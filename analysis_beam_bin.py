"""
Per-beam-bin DBA analysis + modality leave-one-out ablation.
"""
import json, sys, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.special import softmax

sys.path.insert(0, '.')
from config_seq import GlobalConfig
from model_v5 import TransFuserV5
from model2_seq import TransFuser
from data2_seq import CARLA_Data


def compute_DBA_score(y_pred, y_true, max_k=3, delta=5):
    n = y_pred.shape[0]
    yk = np.zeros(max_k)
    for k in range(max_k):
        acc = 0
        idxs = np.arange(k + 1)
        for i in range(n):
            aux1 = np.abs(y_pred[i, idxs] - y_true[i]) / delta
            acc += np.min(np.minimum(aux1, 1.0))
        yk[k] = 1 - acc / n
    return np.mean(yk)


def compute_acc(y_pred, y_true, top_k=[1, 2, 3]):
    total_hits = np.zeros(len(top_k))
    for i in range(len(y_true)):
        for k_idx, k in enumerate(top_k):
            if np.any(y_pred[i, :k] == y_true[i]):
                total_hits[k_idx] += 1
    return np.round(total_hits / len(y_true) * 100, 4)


device = torch.device('cuda:0')

# ── Config ────────────────────────────────────────────────────────────────────
config_v5 = GlobalConfig()
config_v5.add_velocity = 1; config_v5.add_mask = 0; config_v5.enhanced = 1
config_v5.angle_norm = 1; config_v5.custom_FoV_lidar = 1
config_v5.filtered = 0; config_v5.add_seg = 0
config_v5.n_layer = 2
config_v5.embd_pdrop = 0.3; config_v5.resid_pdrop = 0.3; config_v5.attn_pdrop = 0.3

data_root = config_v5.data_root + '/Multi_Modal/'
test_set = CARLA_Data(root=data_root, root_csv='scenario32_test_seq.csv',
                      config=config_v5, test=False)
dl_test = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)
print(f'Test samples: {len(test_set)}')

# ── Load v5b model ────────────────────────────────────────────────────────────
model = TransFuserV5(config_v5, device).to(device)
model.load_state_dict(torch.load('log/s32_v5b/best_model.pth',
                                  map_location=device, weights_only=False))
model.eval()

# ── Collect logits with modality masks ───────────────────────────────────────
def get_logits(model, dl, zero_img=False, zero_lid=False, zero_rad=False,
               zero_gps=False, desc='eval'):
    """Run inference; optionally zero out specific modalities."""
    logit_list, gt_list = [], []
    with torch.no_grad():
        for data in tqdm(dl, desc=desc):
            fronts, lidars, radars = [], [], []
            gps = data['gps'].to(device, dtype=torch.float32)
            if zero_gps:
                gps = torch.zeros_like(gps)
            for i in range(config_v5.seq_len):
                f = data['fronts'][i].to(device, dtype=torch.float32)
                l = data['lidars'][i].to(device, dtype=torch.float32)
                r = data['radars'][i].to(device, dtype=torch.float32)
                if zero_img:
                    f = torch.zeros_like(f)
                if zero_lid:
                    l = torch.zeros_like(l)
                if zero_rad:
                    r = torch.zeros_like(r)
                fronts.append(f); lidars.append(l); radars.append(r)
            logits = model.predict(fronts, lidars, radars, gps)
            gt_list.append(data['beamidx'][0].numpy())
            logit_list.append(logits.cpu().numpy())
    return np.concatenate(logit_list, 0), np.concatenate(gt_list)


# Full model
logits_full, gt = get_logits(model, dl_test, desc='full')

# Modality ablations
logits_noimg, _ = get_logits(model, dl_test, zero_img=True, desc='no-img')
logits_nolid, _ = get_logits(model, dl_test, zero_lid=True, desc='no-lidar')
logits_norad, _ = get_logits(model, dl_test, zero_rad=True, desc='no-radar')
logits_nogps, _ = get_logits(model, dl_test, zero_gps=True, desc='no-gps')

# ── Per-beam-bin analysis ─────────────────────────────────────────────────────
print('\n=== Per-beam-bin DBA Analysis ===')
pred_full = np.argsort(-logits_full, axis=1)

# Split into low / mid / high beam bins (0-21, 22-42, 43-63)
bins = [(0, 21, 'low'), (22, 42, 'mid'), (43, 63, 'high')]
bin_results = {}
print(f'Test beam distribution: mean={gt.mean():.1f}, std={gt.std():.1f}')
print(f'  Val beam mean: ~22.9 (estimated), Test beam mean: {gt.mean():.1f}')
print()
print(f'{"Bin":<10} {"N":<6} {"Top-1":<8} {"DBA":<8} {"Beam range"}')
print('-' * 50)
for lo, hi, label in bins:
    mask = (gt >= lo) & (gt <= hi)
    if mask.sum() == 0:
        continue
    g = gt[mask]
    p = pred_full[mask]
    acc = compute_acc(p, g)
    dba = compute_DBA_score(p, g)
    bin_results[label] = {'n': int(mask.sum()), 'top1': float(acc[0]),
                           'top3': float(acc[2]), 'DBA': float(dba),
                           'beam_range': f'{lo}-{hi}',
                           'mean_beam': float(g.mean())}
    print(f'{label:<10} {mask.sum():<6} {acc[0]:<8.2f} {dba:<8.4f} [{lo}-{hi}]')

# ── Modality ablation results ─────────────────────────────────────────────────
print('\n=== Modality Leave-One-Out Ablation ===')
ablation_results = {}
configs = [
    ('full', logits_full, 'All modalities'),
    ('no_img', logits_noimg, 'No camera'),
    ('no_lid', logits_nolid, 'No LiDAR'),
    ('no_rad', logits_norad, 'No radar'),
    ('no_gps', logits_nogps, 'No GPS'),
]
print(f'{"Config":<20} {"Top-1":<8} {"Top-3":<8} {"DBA"}')
print('-' * 50)
for name, lgt, desc in configs:
    pred = np.argsort(-lgt, axis=1)
    acc = compute_acc(pred, gt)
    dba = compute_DBA_score(pred, gt)
    ablation_results[name] = {'top1': float(acc[0]), 'top3': float(acc[2]),
                               'DBA': float(dba), 'desc': desc}
    print(f'{desc:<20} {acc[0]:<8.2f} {acc[2]:<8.2f} {dba:.4f}')

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    'per_beam_bin': bin_results,
    'modality_ablation': ablation_results,
    'test_beam_stats': {
        'mean': float(gt.mean()),
        'std': float(gt.std()),
        'min': int(gt.min()),
        'max': int(gt.max()),
    }
}
with open('log/beam_bin_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to log/beam_bin_analysis.json')
