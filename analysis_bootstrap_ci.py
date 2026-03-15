"""
Bootstrap confidence intervals for DBA and top-k accuracy.
Tests all models and the best ensemble.
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


def compute_acc(y_pred, y_true, k=1):
    hits = sum(1 for i in range(len(y_true)) if np.any(y_pred[i, :k] == y_true[i]))
    return hits / len(y_true) * 100


def bootstrap_ci(logits, gt, n_bootstrap=2000, alpha=0.05, seed=42):
    """Paired bootstrap CI for DBA and top-1/top-3."""
    rng = np.random.RandomState(seed)
    n = len(gt)
    pred = np.argsort(-logits, axis=1)

    dba_boot = np.zeros(n_bootstrap)
    top1_boot = np.zeros(n_bootstrap)
    top3_boot = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        dba_boot[b] = compute_DBA_score(pred[idx], gt[idx])
        top1_boot[b] = compute_acc(pred[idx], gt[idx], k=1)
        top3_boot[b] = compute_acc(pred[idx], gt[idx], k=3)

    def ci(arr):
        lo = np.percentile(arr, 100 * alpha / 2)
        hi = np.percentile(arr, 100 * (1 - alpha / 2))
        return float(lo), float(hi), float(np.mean(arr)), float(np.std(arr))

    return {
        'DBA': {'mean': ci(dba_boot)[2], 'std': ci(dba_boot)[3],
                'CI_lo': ci(dba_boot)[0], 'CI_hi': ci(dba_boot)[1]},
        'top1': {'mean': ci(top1_boot)[2], 'std': ci(top1_boot)[3],
                 'CI_lo': ci(top1_boot)[0], 'CI_hi': ci(top1_boot)[1]},
        'top3': {'mean': ci(top3_boot)[2], 'std': ci(top3_boot)[3],
                 'CI_lo': ci(top3_boot)[0], 'CI_hi': ci(top3_boot)[1]},
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
            fronts, lidars, radars = [], [], []
            gps = data['gps'].to(device, dtype=torch.float32)
            for i in range(config_v5.seq_len):
                fronts.append(data['fronts'][i].to(device, dtype=torch.float32))
                lidars.append(data['lidars'][i].to(device, dtype=torch.float32))
                radars.append(data['radars'][i].to(device, dtype=torch.float32))
            if use_forward:
                logits = model(fronts, lidars, radars, gps)
            else:
                logits = model.predict(fronts, lidars, radars, gps)
            gt_list.append(data['beamidx'][0].numpy())
            logit_list.append(logits.cpu().numpy())
    return np.concatenate(logit_list, 0), np.concatenate(gt_list)


# Load all models
print('Loading models...')
all_logits = {}

for name, path in [('v5b_s42', 'log/s32_v5b/best_model.pth'),
                   ('v5b_s123', 'log/s32_v5b_s2/best_model.pth'),
                   ('v5b_s7', 'log/s32_v5b_s3/best_model.pth'),
                   ('v9', 'log/s32_v9/best_model.pth')]:
    m = TransFuserV5(config_v5, device).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    lgt, gt = get_logits(m, dl_test, desc=name)
    all_logits[name] = lgt
    del m; torch.cuda.empty_cache()

m_bl = TransFuser(config_bl, device).to(device)
m_bl.load_state_dict(torch.load('log/s32_run3/best_model_baseline_backup.pth',
                                 map_location=device, weights_only=False))
lgt_bl, _ = get_logits(m_bl, dl_test, use_forward=True, desc='baseline')
all_logits['baseline'] = lgt_bl
del m_bl; torch.cuda.empty_cache()

# Best ensemble
prob_ens = np.mean([softmax(all_logits[n], axis=1)
                    for n in ['v5b_s42', 'v5b_s123', 'v5b_s7', 'v9']], axis=0)

# ── Compute bootstrap CIs ─────────────────────────────────────────────────────
print('\n=== Bootstrap 95% CI (n=2000 resamples) ===')
print(f'{"Model":<25} {"DBA (95% CI)":<30} {"Top-1 (95% CI)"}')
print('-' * 75)

results = {}
models_to_test = list(all_logits.items()) + [('ens_v5b3+v9', prob_ens)]

for name, lgt in models_to_test:
    ci = bootstrap_ci(lgt, gt)
    results[name] = ci
    dba_str = f'{ci["DBA"]["mean"]:.4f} [{ci["DBA"]["CI_lo"]:.4f}, {ci["DBA"]["CI_hi"]:.4f}]'
    t1_str = f'{ci["top1"]["mean"]:.2f}% [{ci["top1"]["CI_lo"]:.2f}, {ci["top1"]["CI_hi"]:.2f}]'
    print(f'{name:<25} {dba_str:<30} {t1_str}')

with open('log/bootstrap_ci.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to log/bootstrap_ci.json')
