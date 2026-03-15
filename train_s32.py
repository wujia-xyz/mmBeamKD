"""
DDP training script for TII TransFuser on Scenario 32.

Launch:
    torchrun --nproc_per_node=2 train_s32.py --id s32_run2 --epochs 150 --batch_size 6
    (batch_size 是每张卡的 batch size，实际 effective batch = batch_size * nproc_per_node)

nohup:
    mkdir -p log/s32_run2
    nohup torchrun --nproc_per_node=2 train_s32.py --id s32_run2 --epochs 150 --batch_size 6 \
        > log/s32_run2/train.log 2>&1 &
    echo $! > log/s32_run2/pid.txt
"""
import argparse
import json
import os
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision

torch.backends.cudnn.benchmark = True

from scheduler import CyclicCosineDecayLR
from config_seq import GlobalConfig
from model2_seq import TransFuser
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
parser.add_argument('--id',                type=str,   default='s32')
parser.add_argument('--epochs',            type=int,   default=150)
parser.add_argument('--lr',                type=float, default=5e-4)
parser.add_argument('--batch_size',        type=int,   default=6,   help='per-GPU batch size')
parser.add_argument('--logdir',            type=str,   default='log')
parser.add_argument('--add_velocity',      type=int,   default=1)
parser.add_argument('--add_mask',          type=int,   default=0)
parser.add_argument('--enhanced',          type=int,   default=1)
parser.add_argument('--filtered',          type=int,   default=0)
parser.add_argument('--loss',              type=str,   default='focal', help='ce or focal')
parser.add_argument('--scheduler',         type=int,   default=1)
parser.add_argument('--load_previous_best',type=int,   default=0)
parser.add_argument('--finetune_from',     type=str,   default='', help='path to pretrained model.pth for fine-tuning')
parser.add_argument('--temp_coef',         type=int,   default=1)
parser.add_argument('--angle_norm',        type=int,   default=1)
parser.add_argument('--custom_FoV_lidar',  type=int,   default=1)
parser.add_argument('--add_seg',           type=int,   default=0)
parser.add_argument('--ema',               type=int,   default=0)
parser.add_argument('--flip',              type=int,   default=0)
args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)

# TensorBoard 只在 rank 0 写
if rank == 0:
    writer = SummaryWriter(log_dir=args.logdir)


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, input, target):
        if len(target.shape) == 1:
            target = nn.functional.one_hot(target, num_classes=64)
        return torchvision.ops.sigmoid_focal_loss(
            input, target.float(), alpha=self.alpha, gamma=self.gamma, reduction='mean')


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
    """
    收集所有 rank 的预测结果，截断到 dataset_size（去掉 DistributedSampler 的 padding）。
    rank 0 返回完整数组，其他 rank 返回 None。
    """
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
        self.criterion     = FocalLoss() if args.loss == 'focal' else nn.CrossEntropyLoss(reduction='mean')

    def train(self):
        model.train()
        sampler_train.set_epoch(self.cur_epoch)   # DDP 必须每 epoch 更新 sampler seed
        loss_epoch  = 0.0
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

            pred_beams = model(fronts, lidars, radars, gps)
            gt_beamidx = data['beamidx'][0].to(device, dtype=torch.long)
            gt_beams   = data['beam'][0].to(device, dtype=torch.float32)
            loss = self.criterion(pred_beams, gt_beams if args.temp_coef else gt_beamidx)

            gt_list.append(data['beamidx'][0].numpy())
            pred_list.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())

            loss.backward()
            optimizer.step()
            if args.ema:
                ema.update()

            loss_epoch  += loss.item()
            num_batches += 1
            pbar.set_description(f'loss={loss.item():.4f}')
            self.cur_iter += 1

        # 跨卡平均 loss
        loss_tensor = torch.tensor(loss_epoch / num_batches, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

        # 汇总所有卡的预测，rank 0 计算指标
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
        val_loss    = 0.0
        num_batches = 0

        with torch.no_grad():
            for data in tqdm(dataloader_val, desc='val', disable=(rank != 0)):
                fronts, lidars, radars = [], [], []
                gps = data['gps'].to(device, dtype=torch.float32)
                for i in range(config.seq_len):
                    fronts.append(data['fronts'][i].to(device, dtype=torch.float32))
                    lidars.append(data['lidars'][i].to(device, dtype=torch.float32))
                    radars.append(data['radars'][i].to(device, dtype=torch.float32))

                pred_beams = model(fronts, lidars, radars, gps)
                gt_beamidx = data['beamidx'][0].to(device, dtype=torch.long)
                gt_beams   = data['beam'][0].to(device, dtype=torch.float32)
                loss = self.criterion(pred_beams, gt_beams if args.temp_coef else gt_beamidx)

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
            self.DBA.append(0.0)   # 非 rank 0 的占位，不实际使用

        if args.ema:
            ema.restore()

    def save(self):
        if rank != 0:
            return
        save_best = self.DBA[-1] >= self.bestval
        if save_best:
            self.bestval       = self.DBA[-1]
            self.bestval_epoch = self.cur_epoch

        log_table = {
            'epoch':         self.cur_epoch,
            'iter':          self.cur_iter,
            'bestval':       self.bestval,
            'bestval_epoch': self.bestval_epoch,
            'train_loss':    self.train_loss,
            'val_loss':      self.val_loss,
            'DBA':           self.DBA,
        }
        # 存 model.module.state_dict()，不带 DDP 的 'module.' 前缀，方便后续非 DDP 加载
        torch.save(model.module.state_dict(), os.path.join(args.logdir, 'final_model.pth'))
        with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))

        if save_best:
            torch.save(model.module.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
            torch.save(optimizer.state_dict(),    os.path.join(args.logdir, 'best_optim.pth'))
            tqdm.write(f'====== New best model (DBA={self.bestval:.4f}) ======>')
        elif args.load_previous_best:
            model.module.load_state_dict(
                torch.load(os.path.join(args.logdir, 'best_model.pth'), map_location=device))
            optimizer.load_state_dict(
                torch.load(os.path.join(args.logdir, 'best_optim.pth')))
            tqdm.write('====== Loaded previous best model ======>')


# ── 随机种子（各 rank 不同，保证数据增强多样性）──────────────────────────────────
seed = 100 + rank
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

# ── Data ──────────────────────────────────────────────────────────────────────
data_root  = config.data_root
train_root = data_root + '/Multi_Modal/'

if rank == 0:
    print('Loading datasets...')

train_set = CARLA_Data(root=train_root, root_csv='scenario32_train_seq.csv', config=config, test=False)
val_set   = CARLA_Data(root=train_root, root_csv='scenario32_val_seq.csv',   config=config, test=False)
test_set  = CARLA_Data(root=train_root, root_csv='scenario32_test_seq.csv',  config=config, test=False)

if rank == 0:
    print(f'train:{len(train_set)}, val:{len(val_set)}, test:{len(test_set)}')
    print(f'effective batch size: {args.batch_size * world_size}')

sampler_train = DistributedSampler(train_set, shuffle=True)
sampler_val   = DistributedSampler(val_set,   shuffle=False)
sampler_test  = DistributedSampler(test_set,  shuffle=False)

dataloader_train = DataLoader(
    train_set, batch_size=args.batch_size, sampler=sampler_train,
    num_workers=8, pin_memory=True, worker_init_fn=seed_worker, generator=g)
dataloader_val = DataLoader(
    val_set, batch_size=args.batch_size, sampler=sampler_val,
    num_workers=8, pin_memory=False)
dataloader_test = DataLoader(
    test_set, batch_size=args.batch_size, sampler=sampler_test,
    num_workers=8, pin_memory=False)

# ── Model ─────────────────────────────────────────────────────────────────────
model = TransFuser(config, device).to(device)
model = DDP(model, device_ids=[local_rank])

optimizer = optim.AdamW(model.parameters(), lr=args.lr)

if args.scheduler:
    scheduler = CyclicCosineDecayLR(
        optimizer, init_decay_epochs=15, min_decay_lr=2.5e-6,
        restart_interval=10, restart_lr=12.5e-5,
        warmup_epochs=10, warmup_start_lr=2.5e-6)

trainer = Engine()

if rank == 0:
    params = sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {params:,}')

# ── Logdir / Resume（所有 rank 都读，保证 cur_epoch 同步）────────────────────────
if rank == 0 and not os.path.isdir(args.logdir):
    os.makedirs(args.logdir)
dist.barrier()   # 确保 logdir 已创建

if os.path.isfile(os.path.join(args.logdir, 'recent.log')):
    if rank == 0:
        print(f'Resuming from {args.logdir}')
    with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
        log_table = json.load(f)
    trainer.cur_epoch  = log_table['epoch']
    trainer.bestval    = log_table['bestval']
    trainer.train_loss = log_table['train_loss']
    trainer.val_loss   = log_table['val_loss']
    trainer.DBA        = log_table['DBA']
    model.module.load_state_dict(
        torch.load(os.path.join(args.logdir, 'best_model.pth'),
                   map_location=device, weights_only=False))
elif args.finetune_from:
    if rank == 0:
        print(f'Fine-tuning from {args.finetune_from}')
    model.module.load_state_dict(
        torch.load(args.finetune_from, map_location=device, weights_only=False))
elif rank == 0:
    print(f'Created dir: {args.logdir}')

# EMA 作用在 model.module（非 DDP 包装层）
ema = EMA(model.module, 0.999)
if args.ema:
    ema.register()

if rank == 0:
    with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

# ── Train Loop ────────────────────────────────────────────────────────────────
for epoch in range(trainer.cur_epoch, args.epochs):
    if rank == 0:
        print(f'\n=== Epoch {epoch} ===')
    trainer.train()
    trainer.validate()
    trainer.save()
    dist.barrier()   # 等 rank 0 存完模型，再进下一 epoch
    if args.scheduler:
        if rank == 0:
            print(f'lr: {scheduler.get_lr()}')
        scheduler.step()

# ── Test Evaluation（用 best_model）──────────────────────────────────────────
if rank == 0:
    print('\n========== Test Evaluation ==========')

# 所有 rank 加载 best model
model.module.load_state_dict(
    torch.load(os.path.join(args.logdir, 'best_model.pth'),
               map_location=device, weights_only=False))
dist.barrier()

model.eval()
gt_list, pred_list = [], []
with torch.no_grad():
    for data in tqdm(dataloader_test, desc='test', disable=(rank != 0)):
        fronts, lidars, radars = [], [], []
        gps = data['gps'].to(device, dtype=torch.float32)
        for i in range(config.seq_len):
            fronts.append(data['fronts'][i].to(device, dtype=torch.float32))
            lidars.append(data['lidars'][i].to(device, dtype=torch.float32))
            radars.append(data['radars'][i].to(device, dtype=torch.float32))
        pred_beams = model(fronts, lidars, radars, gps)
        gt_list.append(data['beamidx'][0].numpy())
        pred_list.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())

local_gt   = np.concatenate(gt_list)
local_pred = np.squeeze(np.concatenate(pred_list, 0))
gt_all, pred_all = gather_predictions(local_gt, local_pred, len(test_set))

if rank == 0:
    acc = compute_acc(pred_all, gt_all)
    dba = compute_DBA_score(pred_all, gt_all)
    print(f'  Test  top-1:{acc[0]:.2f}% top-2:{acc[1]:.2f}% top-3:{acc[2]:.2f}% DBA:{dba:.4f}')
    print(f'  Best val DBA:{trainer.bestval:.4f} (epoch {trainer.bestval_epoch})')
    result = {
        'test_top1':       float(acc[0]),
        'test_top2':       float(acc[1]),
        'test_top3':       float(acc[2]),
        'test_DBA':        float(dba),
        'best_val_DBA':    float(trainer.bestval),
        'best_val_epoch':  trainer.bestval_epoch,
    }
    with open(os.path.join(args.logdir, 'test_results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Results saved to {args.logdir}/test_results.json')

dist.destroy_process_group()
