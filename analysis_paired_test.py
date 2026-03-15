"""
Paired bootstrap significance test: distilled vs baseline DBA.
Also computes delta CIs for all key comparisons.
"""
import json, sys, numpy as np, torch
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


def paired_bootstrap_test(logits_a, logits_b, gt, n_bootstrap=5000, seed=42):
    """
    Paired bootstrap test for DBA(a) - DBA(b).
    Returns: delta_mean, delta_CI, p-value (proportion where delta <= 0)
    """
    rng = np.random.RandomState(seed)
    n = len(gt)
    pred_a = np.argsort(-logits_a, axis=1)
    pred_b = np.argsort(-logits_b, axis=1)

    delta_boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        dba_a = compute_DBA_score(pred_a[idx], gt[idx])
        dba_b = compute_DBA_score(pred_b[idx], gt[idx])
        delta_boot[i] = dba_a - dba_b

    delta_mean = float(np.mean(delta_boot))
    delta_ci_lo = float(np.percentile(delta_boot, 2.5))
    delta_ci_hi = float(np.percentile(delta_boot, 97.5))
    p_value = float(np.mean(delta_boot <= 0))  # H0: delta <= 0

    return {
        'delta_mean': delta_mean,
        'delta_CI_lo': delta_ci_lo,
        'delta_CI_hi': delta_ci_hi,
        'p_value': p_value,
        'significant_at_0.05': p_value < 0.05,
    }


device = torch.device('cuda:0')

config_v5 = GlobalConfig()
config_v5.add_velocity = 1; config_v5.add_mask = 0; config_v5.enhanced = 1
config_v5.angle_norm = 1; config_v5.custom_FoV_lidar = 1
config_v5.filtered = 0; config_v5.add_seg = 0; config_v5.n_layer = 2
config_v5.embd_pdrop = 0.3; config_v5.resid_pdrop = 0.3; config_v5.attn_pdrop = 0.3

config_bl = GlobalConfig()
config_bl.add_velocity = 1; config_bl.add_mask = 0; config_bl.enhanced = 1
config_bl.angle_norm = 1; config_bl.custom_FoV_lidar = 1
config_bl.filtered = 0; config_bl.add_seg = 0

data_root = config_v5.data_root + '/Multi_Modal/'
test_set = CARLA_Data(root=data_root, root_csv='scenario32_test_seq.csv',
                      config=config_v5, test=False)
dl_test = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)
print(f'Test: {len(test_set)} samples')


def get_logits(model, dl, use_forward=False, desc=''):
    model.eval()
    gt_list, logit_list = [], []
    with torch.no_grad():
        for data in tqdm(dl, desc=desc, leave=False):
            fronts = [data['fronts'][i].to(device, dtype=torch.float32)
                      for i in range(config_v5.seq_len)]
            lidars = [data['lidars'][i].to(device, dtype=torch.float32)
                      for i in range(config_v5.seq_len)]
            radars = [data['radars'][i].to(device, dtype=torch.float32)
                      for i in range(config_v5.seq_len)]
            gps = data['gps'].to(device, dtype=torch.float32)
            if use_forward:
                logits = model(fronts, lidars, radars, gps)
            else:
                logits = model.predict(fronts, lidars, radars, gps)
            gt_list.append(data['beamidx'][0].numpy())
            logit_list.append(logits.cpu().numpy())
    return np.concatenate(logit_list, 0), np.concatenate(gt_list)


# Load models
print('Loading models...')

m_distill = TransFuserV5(config_v5, device).to(device)
m_distill.load_state_dict(torch.load('log/s32_v10_distill/best_model.pth',
                                      map_location=device, weights_only=False))
logits_distill, gt = get_logits(m_distill, dl_test, desc='distill')
del m_distill; torch.cuda.empty_cache()

m_v5b = TransFuserV5(config_v5, device).to(device)
m_v5b.load_state_dict(torch.load('log/s32_v5b/best_model.pth',
                                   map_location=device, weights_only=False))
logits_v5b, _ = get_logits(m_v5b, dl_test, desc='v5b')
del m_v5b; torch.cuda.empty_cache()

m_v9 = TransFuserV5(config_v5, device).to(device)
m_v9.load_state_dict(torch.load('log/s32_v9/best_model.pth',
                                  map_location=device, weights_only=False))
logits_v9, _ = get_logits(m_v9, dl_test, desc='v9')
del m_v9; torch.cuda.empty_cache()

m_bl = TransFuser(config_bl, device).to(device)
m_bl.load_state_dict(torch.load('log/s32_run3/best_model_baseline_backup.pth',
                                  map_location=device, weights_only=False))
logits_bl, _ = get_logits(m_bl, dl_test, use_forward=True, desc='baseline')
del m_bl; torch.cuda.empty_cache()

# Best ensemble
probs_ens = np.mean([softmax(logits_distill, axis=1),
                     softmax(logits_v5b, axis=1),
                     softmax(logits_v9, axis=1)], axis=0)

print('\n=== Paired Bootstrap Significance Tests (n=5000) ===')
print('H0: model A DBA ≤ baseline DBA (one-sided p-value)')
print()

results = {}

comparisons = [
    ('distill_vs_baseline', logits_distill, logits_bl, 'Distilled vs Baseline'),
    ('v5b_vs_baseline', logits_v5b, logits_bl, 'v5b (s42) vs Baseline'),
    ('ensemble_vs_baseline', probs_ens, logits_bl, 'distill+v5b+v9 ens vs Baseline'),
]

for key, lg_a, lg_b, label in comparisons:
    result = paired_bootstrap_test(lg_a, lg_b, gt)
    results[key] = result
    print(f'{label}:')
    print(f'  DeltaDBA (mean): {result["delta_mean"]:+.4f}')
    print(f'  95% CI: [{result["delta_CI_lo"]:+.4f}, {result["delta_CI_hi"]:+.4f}]')
    print(f'  p-value: {result["p_value"]:.4f} {"(SIGNIFICANT)" if result["significant_at_0.05"] else "(not significant)"}')
    print()

# Checkpoint sizes
import os
sizes = {}
for name, path in [('baseline', 'log/s32_run3/best_model_baseline_backup.pth'),
                   ('v5b', 'log/s32_v5b/best_model.pth'),
                   ('distilled', 'log/s32_v10_distill/best_model.pth')]:
    if os.path.exists(path):
        sizes[name] = os.path.getsize(path) / 1024 / 1024  # MB
        print(f'Checkpoint {name}: {sizes[name]:.1f} MB')

results['checkpoint_sizes_MB'] = sizes

with open('log/paired_significance.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to log/paired_significance.json')
