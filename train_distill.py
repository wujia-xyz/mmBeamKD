"""
Ensemble distillation: train a student TransFuserV5 on ensemble soft labels.

Teacher: 4-model probability-space ensemble (v5b×3 + v9)
Student: TransFuserV5 with same architecture
Loss: KL divergence to teacher probs + ordinal distance loss

Usage:
  torchrun --nproc_per_node=1 --master_port=29603 train_distill.py \
      --id s32_v10_distill --epochs 100 --batch_size 6

Key idea: Student trained on ensemble soft labels can exceed any individual teacher
because the soft labels encode the ensemble's uncertainty and richer signal.
"""
import argparse, json, os, sys, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from scipy.special import softmax as scipy_softmax

sys.path.insert(0, '.')
from config_seq import GlobalConfig
from model_v5 import TransFuserV5, freeze_backbone, FrozenBNMixin
from model2_seq import TransFuser
from data2_seq import CARLA_Data
from losses import FocalLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--id', type=str, default='s32_v10_distill')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--batch_size', type=int, default=6)
    p.add_argument('--logdir', type=str, default=None)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--kd_temp', type=float, default=4.0,
                   help='Temperature for knowledge distillation')
    p.add_argument('--kd_weight', type=float, default=0.7,
                   help='Weight for KD loss (1-kd_weight for hard label loss)')
    p.add_argument('--ordinal_weight', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=99)
    # Model flags
    p.add_argument('--add_velocity', type=int, default=1)
    p.add_argument('--enhanced', type=int, default=1)
    p.add_argument('--angle_norm', type=int, default=1)
    p.add_argument('--custom_FoV_lidar', type=int, default=1)
    return p.parse_args()


def ordinal_distance_loss(logits, gt_idx, delta=5.0):
    bz, n_classes = logits.shape
    probs = F.softmax(logits, dim=-1)
    beam_indices = torch.arange(n_classes, device=logits.device).float()
    distances = torch.abs(beam_indices.unsqueeze(0) - gt_idx.unsqueeze(1).float())
    distances = torch.clamp(distances / delta, max=1.0)
    return (probs * distances).sum(dim=-1).mean()


def main():
    args = parse_args()
    if args.logdir is None:
        args.logdir = f'log/{args.id}'
    os.makedirs(args.logdir, exist_ok=True)

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        dist.init_process_group('nccl')

    seed = args.seed + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(f'cuda:{local_rank}')

    # ── Config ────────────────────────────────────────────────────────────────
    config = GlobalConfig()
    config.add_velocity = args.add_velocity
    config.add_mask = 0; config.enhanced = args.enhanced
    config.angle_norm = args.angle_norm
    config.custom_FoV_lidar = args.custom_FoV_lidar
    config.filtered = 0; config.add_seg = 0
    config.n_layer = 2
    config.embd_pdrop = 0.3; config.resid_pdrop = 0.3; config.attn_pdrop = 0.3

    # ── Data ──────────────────────────────────────────────────────────────────
    data_root = config.data_root + '/Multi_Modal/'
    train_set = CARLA_Data(root=data_root, root_csv='scenario32_train_seq.csv',
                           config=config, test=False)
    val_set = CARLA_Data(root=data_root, root_csv='scenario32_val_seq.csv',
                         config=config, test=False)
    test_set = CARLA_Data(root=data_root, root_csv='scenario32_test_seq.csv',
                          config=config, test=False)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank,
                                       shuffle=True) if world_size > 1 else None
    dl_train = DataLoader(train_set, batch_size=args.batch_size,
                          sampler=train_sampler,
                          shuffle=(train_sampler is None),
                          num_workers=4, pin_memory=True, drop_last=True)
    dl_val = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)
    dl_test = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)

    if rank == 0:
        print(f'Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}')

    # ── Load teacher models (only on GPU 0 for inference) ─────────────────────
    if rank == 0:
        print('Loading teacher ensemble...')
        teacher_models = []
        for path in ['log/s32_v5b/best_model.pth',
                     'log/s32_v5b_s2/best_model.pth',
                     'log/s32_v5b_s3/best_model.pth',
                     'log/s32_v9/best_model.pth']:
            m = TransFuserV5(config, device).to(device)
            m.load_state_dict(torch.load(path, map_location=device, weights_only=False))
            m.eval()
            teacher_models.append(m)
        print(f'Loaded {len(teacher_models)} teacher models')

    # ── Student model ─────────────────────────────────────────────────────────
    student = TransFuserV5(config, device).to(device)
    # Initialize student from v5b seed 42 (best single model)
    student.load_state_dict(torch.load('log/s32_v5b/best_model.pth',
                                        map_location=device, weights_only=False))
    freeze_backbone(student, 'all')

    if world_size > 1:
        student = DDP(student, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    focal_loss = FocalLoss(gamma=2.0)

    # ── Helper functions ──────────────────────────────────────────────────────
    def compute_DBA(y_pred, y_true, max_k=3, delta=5):
        n = y_pred.shape[0]
        yk = np.zeros(max_k)
        for k in range(max_k):
            acc = 0
            idxs = np.arange(k + 1)
            for i in range(n):
                aux1 = np.abs(y_pred[i, idxs] - y_true[i]) / delta
                acc += np.min(np.minimum(aux1, 1.0))
            yk[k] = 1 - acc / n
        return float(np.mean(yk))

    def compute_acc(y_pred, y_true):
        top1 = sum(1 for i in range(len(y_true)) if y_pred[i, 0] == y_true[i])
        return top1 / len(y_true) * 100

    def get_teacher_probs(fronts, lidars, radars, gps):
        """Get ensemble soft labels from teacher models."""
        with torch.no_grad():
            probs_list = []
            for tm in teacher_models:
                logits = tm.predict(fronts, lidars, radars, gps)
                probs_list.append(F.softmax(logits, dim=-1))
            return torch.stack(probs_list, dim=0).mean(dim=0)  # [B, 64]

    def evaluate(model, dl):
        model.eval()
        gt_list, pred_list = [], []
        with torch.no_grad():
            for data in dl:
                fronts = [data['fronts'][i].to(device, dtype=torch.float32)
                          for i in range(config.seq_len)]
                lidars = [data['lidars'][i].to(device, dtype=torch.float32)
                          for i in range(config.seq_len)]
                radars = [data['radars'][i].to(device, dtype=torch.float32)
                          for i in range(config.seq_len)]
                gps = data['gps'].to(device, dtype=torch.float32)
                m = model.module if hasattr(model, 'module') else model
                logits = m.predict(fronts, lidars, radars, gps)
                pred = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
                gt_list.append(data['beamidx'][0].numpy())
                pred_list.append(pred)
        gt = np.concatenate(gt_list)
        pred = np.concatenate(pred_list)
        dba = compute_DBA(pred, gt)
        top1 = compute_acc(pred, gt)
        return dba, top1

    # ── Training ─────────────────────────────────────────────────────────────
    best_val_dba = 0.0
    best_val_epoch = -1
    patience_count = 0

    if rank == 0:
        with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        student.train()
        m = student.module if hasattr(student, 'module') else student
        FrozenBNMixin.freeze_bn_eval(m)

        pbar = tqdm(dl_train, desc=f'Ep{epoch}', disable=(rank != 0))
        for data in pbar:
            fronts = [data['fronts'][i].to(device, dtype=torch.float32)
                      for i in range(config.seq_len)]
            lidars = [data['lidars'][i].to(device, dtype=torch.float32)
                      for i in range(config.seq_len)]
            radars = [data['radars'][i].to(device, dtype=torch.float32)
                      for i in range(config.seq_len)]
            gps = data['gps'].to(device, dtype=torch.float32)
            gt_beamidx = data['beamidx'][0].to(device, dtype=torch.long)

            # Student forward
            ms = student.module if hasattr(student, 'module') else student
            student_logits = ms.predict(fronts, lidars, radars, gps)

            # Teacher soft labels (only on rank 0)
            if rank == 0:
                with torch.no_grad():
                    teacher_probs = get_teacher_probs(fronts, lidars, radars, gps)
            else:
                teacher_probs = torch.zeros_like(student_logits)

            if world_size > 1:
                dist.broadcast(teacher_probs, src=0)

            # KD loss: KL(student || teacher)
            T = args.kd_temp
            student_log_probs_T = F.log_softmax(student_logits / T, dim=-1)
            teacher_probs_T = (teacher_probs ** (1 / T))
            teacher_probs_T = teacher_probs_T / teacher_probs_T.sum(dim=-1, keepdim=True)
            kd_loss = F.kl_div(student_log_probs_T, teacher_probs_T,
                                reduction='batchmean') * (T ** 2)

            # Hard label focal loss (Gaussian soft labels from dataloader)
            gt_soft = data['beam'][0].to(device, dtype=torch.float32)
            hard_loss = focal_loss(student_logits, gt_soft)

            # Ordinal distance loss
            ord_loss = ordinal_distance_loss(student_logits, gt_beamidx)

            loss = (args.kd_weight * kd_loss +
                    (1 - args.kd_weight) * hard_loss +
                    args.ordinal_weight * ord_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            if rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                  'kd': f'{kd_loss.item():.4f}'})

        scheduler.step()

        if rank == 0:
            ms = student.module if hasattr(student, 'module') else student
            val_dba, val_top1 = evaluate(ms, dl_val)
            test_dba, test_top1 = evaluate(ms, dl_test)
            print(f'Epoch {epoch}: Val DBA={val_dba:.4f} top1={val_top1:.2f}% '
                  f'| Test DBA={test_dba:.4f} top1={test_top1:.2f}%')

            if val_dba > best_val_dba:
                best_val_dba = val_dba
                best_val_epoch = epoch
                patience_count = 0
                torch.save(ms.state_dict(),
                           os.path.join(args.logdir, 'best_model.pth'))
                print(f'  >>> New best (val DBA={val_dba:.4f}, test DBA={test_dba:.4f})')
            else:
                patience_count += 1

            if patience_count >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

    if rank == 0:
        print(f'Best val DBA: {best_val_dba:.4f} at epoch {best_val_epoch}')
        # Final test eval
        ms = student.module if hasattr(student, 'module') else student
        ms.load_state_dict(torch.load(os.path.join(args.logdir, 'best_model.pth'),
                                       map_location=device, weights_only=False))
        test_dba, test_top1 = evaluate(ms, dl_test)
        print(f'Final test: DBA={test_dba:.4f} top1={test_top1:.2f}%')
        with open(os.path.join(args.logdir, 'test_results.json'), 'w') as f:
            json.dump({'test_DBA': test_dba, 'test_top1': test_top1,
                       'best_val_DBA': best_val_dba, 'best_val_epoch': best_val_epoch}, f, indent=2)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
