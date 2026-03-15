"""
Val-Test distribution shift analysis.
Generates:
  figures/shift_beam_dist.pdf    — val vs test beam distribution
  figures/shift_acc_gap.pdf      — val vs test per-beam accuracy gap
"""
import sys, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '.')
from config_seq import GlobalConfig
from model_v5 import TransFuserV5
from data2_seq import CARLA_Data

device = torch.device('cuda:0')
config = GlobalConfig()
config.add_velocity = 1; config.add_mask = 0; config.enhanced = 1
config.angle_norm = 1; config.custom_FoV_lidar = 1
config.filtered = 0; config.add_seg = 0; config.n_layer = 2
config.embd_pdrop = 0.3; config.resid_pdrop = 0.3; config.attn_pdrop = 0.3

data_root = config.data_root + '/Multi_Modal/'
val_set  = CARLA_Data(root=data_root, root_csv='scenario32_val_seq.csv',
                      config=config, test=False)
test_set = CARLA_Data(root=data_root, root_csv='scenario32_test_seq.csv',
                      config=config, test=False)
dl_val  = DataLoader(val_set,  batch_size=8, shuffle=False, num_workers=4)
dl_test = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)

# Collect GT beam indices
def get_gt(dl):
    gt_list = []
    for data in tqdm(dl, desc='collecting GT', leave=False):
        gt_list.append(data['beamidx'][0].numpy())
    return np.concatenate(gt_list)

gt_val  = get_gt(dl_val)
gt_test = get_gt(dl_test)

print(f'Val:  n={len(gt_val)}, mean={gt_val.mean():.1f}, std={gt_val.std():.1f}')
print(f'Test: n={len(gt_test)}, mean={gt_test.mean():.1f}, std={gt_test.std():.1f}')

# ── Figure 1: Val vs Test beam distribution ───────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
bins = np.arange(0, 65, 2)
ax1.hist(gt_val, bins=bins, density=True, alpha=0.6, color='#2196F3', label=f'Val (n={len(gt_val)}, μ={gt_val.mean():.1f})')
ax1.hist(gt_test, bins=bins, density=True, alpha=0.6, color='#F44336', label=f'Test (n={len(gt_test)}, μ={gt_test.mean():.1f})')

# KDE
x = np.linspace(0, 63, 300)
kde_val = gaussian_kde(gt_val, bw_method=0.2)
kde_test = gaussian_kde(gt_test, bw_method=0.2)
ax1.plot(x, kde_val(x), color='#1565C0', linewidth=2.5)
ax1.plot(x, kde_test(x), color='#C62828', linewidth=2.5)
ax1.axvline(gt_val.mean(), color='#1565C0', linestyle='--', alpha=0.7)
ax1.axvline(gt_test.mean(), color='#C62828', linestyle='--', alpha=0.7)
ax1.set_xlabel('Beam Index', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Val vs Test Beam Index Distribution', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Cumulative distribution
sorted_beams = np.arange(0, 64)
val_cdf = np.array([(gt_val <= b).mean() for b in sorted_beams])
test_cdf = np.array([(gt_test <= b).mean() for b in sorted_beams])
ax2.plot(sorted_beams, val_cdf, color='#2196F3', linewidth=2.5, label='Val CDF')
ax2.plot(sorted_beams, test_cdf, color='#F44336', linewidth=2.5, label='Test CDF')
ax2.fill_between(sorted_beams, val_cdf, test_cdf, alpha=0.2, color='gray',
                 label='Distribution Gap')
ax2.set_xlabel('Beam Index', fontsize=11)
ax2.set_ylabel('Cumulative Probability', fontsize=11)
ax2.set_title('Cumulative Distribution Function', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# Max shift annotation
ks_stat = np.max(np.abs(val_cdf - test_cdf))
ks_beam = sorted_beams[np.argmax(np.abs(val_cdf - test_cdf))]
ax2.annotate(f'Max gap: {ks_stat:.3f}\n(at beam {ks_beam})',
             xy=(ks_beam, (val_cdf[ks_beam] + test_cdf[ks_beam]) / 2),
             xytext=(ks_beam + 5, 0.3), fontsize=9,
             arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig('figures/shift_beam_dist.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: figures/shift_beam_dist.pdf')

# ── Figure 2: Per-beam val vs test accuracy gap ────────────────────────────────
# Load distilled model, run on both val and test
m = TransFuserV5(config, device).to(device)
m.load_state_dict(torch.load('log/s32_v10_distill/best_model.pth',
                              map_location=device, weights_only=False))
m.eval()

def get_preds(dl):
    pred_list, gt_list = [], []
    with torch.no_grad():
        for data in tqdm(dl, desc='inference', leave=False):
            fronts = [data['fronts'][i].to(device, dtype=torch.float32)
                      for i in range(config.seq_len)]
            lidars = [data['lidars'][i].to(device, dtype=torch.float32)
                      for i in range(config.seq_len)]
            radars = [data['radars'][i].to(device, dtype=torch.float32)
                      for i in range(config.seq_len)]
            gps = data['gps'].to(device, dtype=torch.float32)
            pred_list.append(np.argmax(m.predict(fronts, lidars, radars, gps).cpu().numpy(), axis=1))
            gt_list.append(data['beamidx'][0].numpy())
    return np.concatenate(pred_list), np.concatenate(gt_list)

preds_val, gt_val2 = get_preds(dl_val)
preds_test, gt_test2 = get_preds(dl_test)
del m; torch.cuda.empty_cache()

# Per-beam accuracy
val_acc, test_acc, beam_labels = [], [], []
for b in range(64):
    mask_v = gt_val2 == b
    mask_t = gt_test2 == b
    if mask_v.sum() > 0 and mask_t.sum() > 0:
        val_acc.append((preds_val[mask_v] == b).mean() * 100)
        test_acc.append((preds_test[mask_t] == b).mean() * 100)
        beam_labels.append(b)

val_acc = np.array(val_acc)
test_acc = np.array(test_acc)
gap = val_acc - test_acc  # positive = val better than test

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Top: side-by-side bar
x = np.array(beam_labels)
axes[0].bar(x - 0.4, val_acc, 0.8, alpha=0.7, color='#2196F3', label='Val accuracy')
axes[0].bar(x + 0.4, test_acc, 0.8, alpha=0.7, color='#F44336', label='Test accuracy')
axes[0].set_ylabel('Top-1 Accuracy (%)', fontsize=11)
axes[0].set_title('Per-Beam Accuracy: Val vs Test (Distilled Model)', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# Bottom: gap bar (val − test)
colors = ['#2196F3' if g >= 0 else '#F44336' for g in gap]
axes[1].bar(x, gap, color=colors, alpha=0.7, width=0.8)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].axhline(gap.mean(), color='orange', linestyle='--',
                label=f'Mean gap: {gap.mean():.1f}%')
axes[1].set_xlabel('Beam Index', fontsize=11)
axes[1].set_ylabel('Val − Test Accuracy (%)', fontsize=11)
axes[1].set_title('Per-Beam Accuracy Gap (Positive = Val Overestimates Test)', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/shift_acc_gap.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: figures/shift_acc_gap.pdf')
print('\nAll distribution shift figures saved.')
