"""
TransFuser-v5: Lightweight cross-attention fusion + contrastive alignment.

Key innovations vs baseline:
1. Frozen ResNet backbones → 21.7M trainable (vs 78M)
2. GPT layers: 8→2 per scale
3. Cross-attention fusion token (replaces element-wise sum)
4. InfoNCE contrastive alignment between modalities
5. DBA-aware ordinal loss component
6. Data augmentation (flip)
7. Proper eval protocol (test only at end)

Launch:
    mkdir -p log/s32_v5
    nohup torchrun --nproc_per_node=2 --master_port=29501 train_s32_v5.py \
        --id s32_v5 --epochs 100 --batch_size 6 \
        --finetune_from log/s32_run3/best_model_baseline_backup.pth \
        --freeze_backbone all --n_layer 2 --dropout 0.3 \
        --contrastive_weight 0.1 --ordinal_weight 0.5 \
        --flip_aug 1 --ema 1 --patience 20 \
        --add_velocity 1 --enhanced 1 --angle_norm 1 --custom_FoV_lidar 1 \
        > log/s32_v5/train.log 2>&1 &
    echo $! > log/s32_v5/pid.txt
"""
import argparse
import json
import os
import random
import copy

from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim.swa_utils import AveragedModel, SWALR

torch.backends.cudnn.benchmark = True

from scheduler import CyclicCosineDecayLR
from config_seq import GlobalConfig
from model_v5 import TransFuserV5, freeze_backbone
from data2_seq import CARLA_Data

# ── DDP init ──────────────────────────────────────────────────────────────────
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
rank       = dist.get_rank()
world_size = dist.get_world_size()
device     = torch.device(f'cuda:{local_rank}')
torch.cuda.set_device(device)
torch.cuda.empty_cache()

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--id',                type=str,   default='s32_v5')
parser.add_argument('--epochs',            type=int,   default=100)
parser.add_argument('--lr',                type=float, default=3e-4)
parser.add_argument('--batch_size',        type=int,   default=6,   help='per-GPU batch size')
parser.add_argument('--logdir',            type=str,   default='log')
parser.add_argument('--add_velocity',      type=int,   default=1)
parser.add_argument('--add_mask',          type=int,   default=0)
parser.add_argument('--enhanced',          type=int,   default=1)
parser.add_argument('--filtered',          type=int,   default=0)
parser.add_argument('--temp_coef',         type=int,   default=1)
parser.add_argument('--angle_norm',        type=int,   default=1)
parser.add_argument('--custom_FoV_lidar',  type=int,   default=1)
parser.add_argument('--add_seg',           type=int,   default=0)
parser.add_argument('--finetune_from',     type=str,   default='')
parser.add_argument('--scheduler',         type=int,   default=1)
# v5 specific
parser.add_argument('--n_layer',           type=int,   default=2,   help='GPT layers per scale (baseline=8)')
parser.add_argument('--dropout',           type=float, default=0.3)
parser.add_argument('--freeze_backbone',   type=str,   default='all', choices=['none','early','all'])
parser.add_argument('--contrastive_weight',type=float, default=0.1, help='InfoNCE loss weight')
parser.add_argument('--ordinal_weight',    type=float, default=0.5, help='ordinal distance loss weight')
parser.add_argument('--flip_aug',          type=int,   default=1,   help='enable flip augmentation')
parser.add_argument('--ema',               type=int,   default=1)
parser.add_argument('--ema_decay',         type=float, default=0.999)
parser.add_argument('--patience',          type=int,   default=20)
parser.add_argument('--swa_start',         type=int,   default=60)
parser.add_argument('--swa_lr',            type=float, default=1e-5)
parser.add_argument('--label_smooth',      type=float, default=0.1)
parser.add_argument('--seed',              type=int,   default=42)
args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)

if rank == 0:
    writer = SummaryWriter(log_dir=args.logdir)


# ── Loss Functions ────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, input, target):
        if len(target.shape) == 1:
            target = F.one_hot(target, num_classes=64)
        return torchvision.ops.sigmoid_focal_loss(
            input, target.float(), alpha=self.alpha, gamma=self.gamma, reduction='mean')


def smooth_labels(targets, n_classes=64, smoothing=0.1):
    """Apply label smoothing to soft targets."""
    with torch.no_grad():
        uniform = torch.ones_like(targets) / n_classes
        return (1 - smoothing) * targets + smoothing * uniform


def ordinal_distance_loss(logits, gt_idx, delta=5.0):
    """
    DBA-aligned ordinal loss: penalizes predictions proportional to beam distance.
    Encourages the model to rank nearby beams higher even if not exact match.
    """
    bz, n_classes = logits.shape
    probs = F.softmax(logits, dim=-1)  # (B, 64)
    beam_indices = torch.arange(n_classes, device=logits.device).float()  # (64,)
    # Distance of each beam from ground truth
    distances = torch.abs(beam_indices.unsqueeze(0) - gt_idx.unsqueeze(1).float()) / delta  # (B, 64)
    distances = torch.clamp(distances, max=1.0)
    # Expected distance under predicted distribution
    loss = (probs * distances).sum(dim=-1).mean()
    return loss


def info_nce_loss(z1, z2, temperature=0.07):
    """Symmetric InfoNCE loss between two sets of normalized embeddings."""
    bz = z1.size(0)
    if bz <= 1:
        return torch.tensor(0.0, device=z1.device)
    sim = z1 @ z2.T / temperature  # (B, B)
    labels = torch.arange(bz, device=z1.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss


# ── EMA ───────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_acc(y_pred, y_true, top_k=[1, 2, 3]):
    total_hits = np.zeros(len(top_k))
    for i in range(len(y_true)):
        for k_idx, k in enumerate(top_k):
            if np.any(y_pred[i, :k] == y_true[i]):
                total_hits[k_idx] += 1
    return np.round(total_hits / len(y_true) * 100, 4)


def compute_DBA_score(y_pred, y_true, max_k=3, delta=5):
    n_samples = y_pred.shape[0]
    yk = np.zeros(max_k)
    for k in range(max_k):
        acc = 0
        idxs = np.arange(k + 1)
        for i in range(n_samples):
            aux1 = np.abs(y_pred[i, idxs] - y_true[i]) / delta
            acc += np.min(np.minimum(aux1, 1.0))
        yk[k] = 1 - acc / n_samples
    return np.mean(yk)


def gather_predictions(local_gt, local_pred, dataset_size):
    gathered = [None] * world_size
    dist.all_gather_object(gathered, (local_gt, local_pred))
    if rank == 0:
        gt_all   = np.concatenate([g[0] for g in gathered])[:dataset_size]
        pred_all = np.concatenate([g[1] for g in gathered])[:dataset_size]
        return gt_all, pred_all
    return None, None


# ── Engine ────────────────────────────────────────────────────────────────────
class Engine:
    def __init__(self, cur_epoch=0, cur_iter=0):
        self.cur_epoch     = cur_epoch
        self.cur_iter      = cur_iter
        self.bestval_epoch = cur_epoch
        self.train_loss    = []
        self.val_loss      = []
        self.DBA           = []
        self.bestval       = 0
        self.focal_loss    = FocalLoss()
        self.no_improve    = 0  # early stopping counter

    def train(self):
        model.train()
        sampler_train.set_epoch(self.cur_epoch)
        loss_epoch = 0.0
        num_batches = 0
        gt_list, pred_list = [], []

        pbar = tqdm(dataloader_train, desc='train', disable=(rank != 0))
        for data in pbar:
            optimizer.zero_grad(set_to_none=True)
            fronts, lidars, radars = [], [], []
            gps = data['gps'].to(device, dtype=torch.float32)
            for i in range(config.seq_len):
                fronts.append(data['fronts'][i].to(device, dtype=torch.float32))
                lidars.append(data['lidars'][i].to(device, dtype=torch.float32))
                radars.append(data['radars'][i].to(device, dtype=torch.float32))

            beam_logits, z_img, z_lid, z_rad = model(fronts, lidars, radars, gps)
            gt_beamidx = data['beamidx'][0].to(device, dtype=torch.long)
            gt_beams   = data['beam'][0].to(device, dtype=torch.float32)

            # Label smoothing
            if args.label_smooth > 0:
                gt_beams = smooth_labels(gt_beams, smoothing=args.label_smooth)

            # Main classification loss
            loss_cls = self.focal_loss(beam_logits, gt_beams if args.temp_coef else gt_beamidx)

            # Ordinal distance loss (DBA-aligned)
            loss_ord = ordinal_distance_loss(beam_logits, gt_beamidx)

            # Contrastive alignment loss (InfoNCE between modality pairs)
            loss_con = (info_nce_loss(z_img, z_lid) +
                        info_nce_loss(z_img, z_rad) +
                        info_nce_loss(z_lid, z_rad)) / 3.0

            loss = loss_cls + args.ordinal_weight * loss_ord + args.contrastive_weight * loss_con

            gt_list.append(data['beamidx'][0].numpy())
            pred_list.append(torch.argsort(beam_logits, dim=1, descending=True).cpu().numpy())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if args.ema:
                ema.update()

            loss_epoch  += loss.item()
            num_batches += 1
            pbar.set_description(f'loss={loss.item():.4f}')
            self.cur_iter += 1

        loss_tensor = torch.tensor(loss_epoch / num_batches, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

        local_gt   = np.concatenate(gt_list)
        local_pred = np.squeeze(np.concatenate(pred_list, 0))
        gt_all, pred_all = gather_predictions(local_gt, local_pred, len(train_set))

        self.train_loss.append(avg_loss)
        self.cur_epoch += 1

        if rank == 0:
            acc = compute_acc(pred_all, gt_all)
            dba = compute_DBA_score(pred_all, gt_all)
            print(f'  Train top-1:{acc[0]:.2f}% top-2:{acc[1]:.2f}% top-3:{acc[2]:.2f}% DBA:{dba:.4f} loss:{avg_loss:.4f}')
            writer.add_scalar('train/DBA',  dba,      self.cur_epoch)
            writer.add_scalar('train/loss', avg_loss, self.cur_epoch)

    def validate(self):
        if args.ema:
            ema.apply_shadow()
        model.eval()
        gt_list, pred_list = [], []
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data in tqdm(dataloader_val, desc='val', disable=(rank != 0)):
                fronts, lidars, radars = [], [], []
                gps = data['gps'].to(device, dtype=torch.float32)
                for i in range(config.seq_len):
                    fronts.append(data['fronts'][i].to(device, dtype=torch.float32))
                    lidars.append(data['lidars'][i].to(device, dtype=torch.float32))
                    radars.append(data['radars'][i].to(device, dtype=torch.float32))

                # Use predict() for inference (no contrastive heads)
                pred_beams = model.module.predict(fronts, lidars, radars, gps)
                gt_beamidx = data['beamidx'][0].to(device, dtype=torch.long)
                gt_beams   = data['beam'][0].to(device, dtype=torch.float32)
                loss = self.focal_loss(pred_beams, gt_beams if args.temp_coef else gt_beamidx)

                gt_list.append(data['beamidx'][0].numpy())
                pred_list.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())
                val_loss    += loss.item()
                num_batches += 1

        loss_tensor = torch.tensor(val_loss / num_batches, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

        local_gt   = np.concatenate(gt_list)
        local_pred = np.squeeze(np.concatenate(pred_list, 0))
        gt_all, pred_all = gather_predictions(local_gt, local_pred, len(val_set))

        self.val_loss.append(avg_loss)
        if rank == 0:
            acc = compute_acc(pred_all, gt_all)
            dba = compute_DBA_score(pred_all, gt_all)
            self.DBA.append(dba)
            print(f'  Val   top-1:{acc[0]:.2f}% top-2:{acc[1]:.2f}% top-3:{acc[2]:.2f}% DBA:{dba:.4f} loss:{avg_loss:.4f}')
            writer.add_scalar('val/DBA',  dba,      self.cur_epoch)
            writer.add_scalar('val/loss', avg_loss, self.cur_epoch)
        else:
            self.DBA.append(0.0)

        if args.ema:
            ema.restore()

    def save(self):
        if rank != 0:
            return
        save_best = self.DBA[-1] >= self.bestval
        if save_best:
            self.bestval       = self.DBA[-1]
            self.bestval_epoch = self.cur_epoch
            self.no_improve    = 0
        else:
            self.no_improve += 1

        log_table = {
            'epoch':         self.cur_epoch,
            'iter':          self.cur_iter,
            'bestval':       self.bestval,
            'bestval_epoch': self.bestval_epoch,
            'train_loss':    self.train_loss,
            'val_loss':      self.val_loss,
            'DBA':           self.DBA,
        }
        torch.save(model.module.state_dict(), os.path.join(args.logdir, 'final_model.pth'))
        with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))

        if save_best:
            torch.save(model.module.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
            torch.save(optimizer.state_dict(),    os.path.join(args.logdir, 'best_optim.pth'))
            tqdm.write(f'====== New best model (DBA={self.bestval:.4f}) ======>')

    def should_stop(self):
        """Early stopping check — broadcast from rank 0."""
        stop = torch.tensor(0, device=device)
        if rank == 0 and self.no_improve >= args.patience:
            stop = torch.tensor(1, device=device)
        dist.broadcast(stop, src=0)
        return stop.item() == 1


# ── Random seed ───────────────────────────────────────────────────────────────
seed = args.seed + rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.use_deterministic_algorithms(False)
g = torch.Generator()
g.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ── Config ────────────────────────────────────────────────────────────────────
config = GlobalConfig()
config.add_velocity    = args.add_velocity
config.add_mask        = args.add_mask
config.enhanced        = args.enhanced
config.angle_norm      = args.angle_norm
config.custom_FoV_lidar= args.custom_FoV_lidar
config.filtered        = args.filtered
config.add_seg         = args.add_seg
# v5 overrides
config.n_layer    = args.n_layer
config.embd_pdrop = args.dropout
config.resid_pdrop= args.dropout
config.attn_pdrop = args.dropout

# ── Data ──────────────────────────────────────────────────────────────────────
data_root  = config.data_root
train_root = data_root + '/Multi_Modal/'

if rank == 0:
    print('Loading datasets...')

train_set = CARLA_Data(root=train_root, root_csv='scenario32_train_seq.csv', config=config, test=False)
val_set   = CARLA_Data(root=train_root, root_csv='scenario32_val_seq.csv',   config=config, test=False)
test_set  = CARLA_Data(root=train_root, root_csv='scenario32_test_seq.csv',  config=config, test=False)

# Flip augmentation: create a flipped copy of training data
if args.flip_aug:
    train_set_flip = CARLA_Data(root=train_root, root_csv='scenario32_train_seq.csv',
                                config=config, test=False, flip=True)
    train_set = ConcatDataset([train_set, train_set_flip])

if rank == 0:
    print(f'train:{len(train_set)} (flip_aug={args.flip_aug}), val:{len(val_set)}, test:{len(test_set)}')
    print(f'effective batch size: {args.batch_size * world_size}')

sampler_train = DistributedSampler(train_set, shuffle=True)
sampler_val   = DistributedSampler(val_set,   shuffle=False)
sampler_test  = DistributedSampler(test_set,  shuffle=False)

dataloader_train = DataLoader(
    train_set, batch_size=args.batch_size, sampler=sampler_train,
    num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
dataloader_val = DataLoader(
    val_set, batch_size=args.batch_size, sampler=sampler_val,
    num_workers=4, pin_memory=False)
dataloader_test = DataLoader(
    test_set, batch_size=args.batch_size, sampler=sampler_test,
    num_workers=4, pin_memory=False)

# ── Model ─────────────────────────────────────────────────────────────────────
model = TransFuserV5(config, device).to(device)

# Load pretrained weights (from baseline) with partial matching
if args.finetune_from:
    if rank == 0:
        print(f'Loading pretrained weights from {args.finetune_from}')
    pretrained = torch.load(args.finetune_from, map_location=device, weights_only=False)
    model_dict = model.state_dict()
    # Match keys that exist in both (skip mismatched shapes)
    matched = {}
    for k, v in pretrained.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            matched[k] = v
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    if rank == 0:
        print(f'  Loaded {len(matched)}/{len(pretrained)} parameters')

# Freeze backbones
if args.freeze_backbone != 'none':
    freeze_backbone(model, freeze_layers=args.freeze_backbone)

model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

# Only optimize trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

if args.scheduler:
    scheduler = CyclicCosineDecayLR(
        optimizer, init_decay_epochs=15, min_decay_lr=2.5e-6,
        restart_interval=10, restart_lr=12.5e-5,
        warmup_epochs=10, warmup_start_lr=2.5e-6)

trainer = Engine()

if rank == 0:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_count:,}')
    print(f'Frozen parameters: {total_params - trainable_count:,}')

# ── Logdir ────────────────────────────────────────────────────────────────────
if rank == 0 and not os.path.isdir(args.logdir):
    os.makedirs(args.logdir)
dist.barrier()

# EMA
ema = EMA(model.module, args.ema_decay)
if args.ema:
    ema.register()

if rank == 0:
    with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

# ── SWA setup ─────────────────────────────────────────────────────────────────
swa_model = AveragedModel(model.module)
swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
swa_started = False

# ── Train Loop ────────────────────────────────────────────────────────────────
for epoch in range(trainer.cur_epoch, args.epochs):
    if rank == 0:
        print(f'\n=== Epoch {epoch} ===')

    trainer.train()
    trainer.validate()
    trainer.save()
    dist.barrier()

    # SWA phase
    if epoch >= args.swa_start:
        if not swa_started:
            if rank == 0:
                print('>>> SWA started <<<')
            swa_started = True
        swa_model.update_parameters(model.module)
        swa_scheduler.step()
    elif args.scheduler:
        if rank == 0:
            print(f'lr: {scheduler.get_lr()}')
        scheduler.step()

    # Early stopping
    if trainer.should_stop():
        if rank == 0:
            print(f'Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)')
        break

# ── Update SWA BN ─────────────────────────────────────────────────────────────
if swa_started:
    if rank == 0:
        print('\nUpdating SWA batch normalization...')
    # Custom BN update for dict-based dataloader
    swa_model.train()
    momenta = {}
    for module in swa_model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
            module.momentum = None
            module.num_batches_tracked *= 0
    with torch.no_grad():
        for data in tqdm(dataloader_train, desc='swa-bn', disable=(rank != 0)):
            fronts, lidars, radars = [], [], []
            gps = data['gps'].to(device, dtype=torch.float32)
            for i in range(config.seq_len):
                fronts.append(data['fronts'][i].to(device, dtype=torch.float32))
                lidars.append(data['lidars'][i].to(device, dtype=torch.float32))
                radars.append(data['radars'][i].to(device, dtype=torch.float32))
            swa_model.module.predict(fronts, lidars, radars, gps)
    for module in momenta:
        module.momentum = momenta[module]
    if rank == 0:
        torch.save(swa_model.module.state_dict(), os.path.join(args.logdir, 'swa_model.pth'))
        print('SWA model saved.')

# ── Test Evaluation ───────────────────────────────────────────────────────────
if rank == 0:
    print('\n========== Test Evaluation ==========')

# Test with best model
model.module.load_state_dict(
    torch.load(os.path.join(args.logdir, 'best_model.pth'),
               map_location=device, weights_only=False))
dist.barrier()

model.eval()
gt_list, pred_list = [], []
with torch.no_grad():
    for data in tqdm(dataloader_test, desc='test-best', disable=(rank != 0)):
        fronts, lidars, radars = [], [], []
        gps = data['gps'].to(device, dtype=torch.float32)
        for i in range(config.seq_len):
            fronts.append(data['fronts'][i].to(device, dtype=torch.float32))
            lidars.append(data['lidars'][i].to(device, dtype=torch.float32))
            radars.append(data['radars'][i].to(device, dtype=torch.float32))
        pred_beams = model.module.predict(fronts, lidars, radars, gps)
        gt_list.append(data['beamidx'][0].numpy())
        pred_list.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())

local_gt   = np.concatenate(gt_list)
local_pred = np.squeeze(np.concatenate(pred_list, 0))
gt_all, pred_all = gather_predictions(local_gt, local_pred, len(test_set))

if rank == 0:
    acc = compute_acc(pred_all, gt_all)
    dba = compute_DBA_score(pred_all, gt_all)
    print(f'  [Best] Test top-1:{acc[0]:.2f}% top-2:{acc[1]:.2f}% top-3:{acc[2]:.2f}% DBA:{dba:.4f}')
    result = {
        'best_test_top1': float(acc[0]), 'best_test_top2': float(acc[1]),
        'best_test_top3': float(acc[2]), 'best_test_DBA': float(dba),
        'best_val_DBA': float(trainer.bestval), 'best_val_epoch': trainer.bestval_epoch,
    }

# Test with SWA model if available
if swa_started:
    swa_model.eval()
    gt_list, pred_list = [], []
    with torch.no_grad():
        for data in tqdm(dataloader_test, desc='test-swa', disable=(rank != 0)):
            fronts, lidars, radars = [], [], []
            gps = data['gps'].to(device, dtype=torch.float32)
            for i in range(config.seq_len):
                fronts.append(data['fronts'][i].to(device, dtype=torch.float32))
                lidars.append(data['lidars'][i].to(device, dtype=torch.float32))
                radars.append(data['radars'][i].to(device, dtype=torch.float32))
            pred_beams = swa_model.module.predict(fronts, lidars, radars, gps)
            gt_list.append(data['beamidx'][0].numpy())
            pred_list.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())

    local_gt   = np.concatenate(gt_list)
    local_pred = np.squeeze(np.concatenate(pred_list, 0))
    gt_all, pred_all = gather_predictions(local_gt, local_pred, len(test_set))

    if rank == 0:
        acc_swa = compute_acc(pred_all, gt_all)
        dba_swa = compute_DBA_score(pred_all, gt_all)
        print(f'  [SWA]  Test top-1:{acc_swa[0]:.2f}% top-2:{acc_swa[1]:.2f}% top-3:{acc_swa[2]:.2f}% DBA:{dba_swa:.4f}')
        result['swa_test_top1'] = float(acc_swa[0])
        result['swa_test_top2'] = float(acc_swa[1])
        result['swa_test_top3'] = float(acc_swa[2])
        result['swa_test_DBA']  = float(dba_swa)

if rank == 0:
    with open(os.path.join(args.logdir, 'test_results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Results saved to {args.logdir}/test_results.json')

dist.destroy_process_group()
