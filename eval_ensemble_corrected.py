"""
Corrected ensemble evaluation using clean seeds (s2b/s3b with flip_aug=0).
Run after s2b and s3b training completes.
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


def compute_acc(y_pred, y_true, top_k=[1, 2, 3]):
    total_hits = np.zeros(len(top_k))
    for i in range(len(y_true)):
        for k_idx, k in enumerate(top_k):
            if np.any(y_pred[i, :k] == y_true[i]):
                total_hits[k_idx] += 1
    return np.round(total_hits / len(y_true) * 100, 4)


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


def bootstrap_ci(logits, gt, n=2000, seed=42):
    rng = np.random.RandomState(seed)
    pred = np.argsort(-logits, axis=1)
    n_samples = len(gt)
    dba_boot = []
    for _ in range(n):
        idx = rng.randint(0, n_samples, n_samples)
        dba_boot.append(compute_DBA_score(pred[idx], gt[idx]))
    dba_boot = np.array(dba_boot)
    return {'mean': float(dba_boot.mean()),
            'CI_lo': float(np.percentile(dba_boot, 2.5)),
            'CI_hi': float(np.percentile(dba_boot, 97.5))}


def eval_logits(logits, gt, label=""):
    pred = np.argsort(-logits, axis=1)
    acc = compute_acc(pred, gt)
    dba = compute_DBA_score(pred, gt)
    ci = bootstrap_ci(logits, gt)
    print(f'  [{label}] top-1:{acc[0]:.2f}% top-3:{acc[2]:.2f}% '
          f'DBA:{dba:.4f} [95% CI: {ci["CI_lo"]:.4f}, {ci["CI_hi"]:.4f}]')
    return {'top1': float(acc[0]), 'top2': float(acc[1]), 'top3': float(acc[2]),
            'DBA': float(dba), 'DBA_CI_lo': ci['CI_lo'], 'DBA_CI_hi': ci['CI_hi']}


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


# ── Load all models ───────────────────────────────────────────────────────────
print('Loading models...')
all_logits = {}
gt = None

# Corrected seeds (with flip_aug=0, epochs=100)
seed_models = {}
for seed, name, path in [
    (42,  'v5b_s42',  'log/s32_v5b/best_model.pth'),        # original clean
    (123, 'v5b_s123_clean', 'log/s32_v5b_s2b/best_model.pth'),  # corrected
    (7,   'v5b_s7_clean',   'log/s32_v5b_s3b/best_model.pth'),  # corrected
]:
    import os
    if not os.path.exists(path):
        print(f'  SKIP {name}: {path} not found')
        continue
    m = TransFuserV5(config_v5, device).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    lgt, gt_tmp = get_logits(m, dl_test, desc=name)
    all_logits[name] = lgt
    if gt is None:
        gt = gt_tmp
    del m; torch.cuda.empty_cache()

# v9 (frozen BN fix)
m = TransFuserV5(config_v5, device).to(device)
m.load_state_dict(torch.load('log/s32_v9/best_model.pth',
                              map_location=device, weights_only=False))
lgt, _ = get_logits(m, dl_test, desc='v9')
all_logits['v9'] = lgt
del m; torch.cuda.empty_cache()

# Baseline
m = TransFuser(config_bl, device).to(device)
m.load_state_dict(torch.load('log/s32_run3/best_model_baseline_backup.pth',
                               map_location=device, weights_only=False))
lgt, _ = get_logits(m, dl_test, use_forward=True, desc='baseline')
all_logits['baseline'] = lgt
del m; torch.cuda.empty_cache()

# Distilled model (if available)
dist_path = 'log/s32_v10_distill/best_model.pth'
import os
if os.path.exists(dist_path):
    m = TransFuserV5(config_v5, device).to(device)
    m.load_state_dict(torch.load(dist_path, map_location=device, weights_only=False))
    lgt, _ = get_logits(m, dl_test, desc='distilled')
    all_logits['distilled'] = lgt
    del m; torch.cuda.empty_cache()

# ── Evaluate ─────────────────────────────────────────────────────────────────
results = {}

print('\n=== Individual Models ===')
for name, logits in all_logits.items():
    results[name] = eval_logits(logits, gt, name)

print('\n=== Probability-Space Ensembles ===')


def prob_ensemble(names, label):
    avail = [n for n in names if n in all_logits]
    if len(avail) < 2:
        print(f'  [{label}] Skipped (only {len(avail)} models available)')
        return None
    probs = [softmax(all_logits[n], axis=1) for n in avail]
    avg = np.mean(probs, axis=0)
    return eval_logits(avg, gt, label)


# Clean 3-seed ensemble (all flip_aug=0, same config)
clean_seeds = ['v5b_s42', 'v5b_s123_clean', 'v5b_s7_clean']
r = prob_ensemble(clean_seeds, 'v5b×3seeds_clean')
if r:
    results['v5b_3seed_clean'] = r

# Clean 3-seed + v9
r = prob_ensemble(clean_seeds + ['v9'], 'v5b×3seeds_clean+v9')
if r:
    results['v5b_3seed_clean_v9'] = r

# All models
r = prob_ensemble(list(all_logits.keys()), 'all_models')
if r:
    results['all_models'] = r

# Distilled model ensemble (if available)
if 'distilled' in all_logits:
    r = prob_ensemble(['distilled', 'v5b_s42', 'v9'], 'distill+v5b+v9')
    if r:
        results['distill_ens'] = r

# Save
with open('log/ensemble_corrected_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to log/ensemble_corrected_results.json')
