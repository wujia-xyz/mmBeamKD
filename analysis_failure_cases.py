"""
Failure case analysis for distilled model.
Generates:
  figures/failure_error_dist.pdf  — error distribution by beam bin
  figures/failure_confusion.pdf   — beam confusion heatmap
  figures/failure_per_beam.pdf    — per-beam top-1 accuracy
"""
import sys, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
test_set = CARLA_Data(root=data_root, root_csv='scenario32_test_seq.csv',
                      config=config, test=False)
dl = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)

# Load distilled model
m = TransFuserV5(config, device).to(device)
m.load_state_dict(torch.load('log/s32_v10_distill/best_model.pth',
                              map_location=device, weights_only=False))
m.eval()

logit_list, gt_list = [], []
with torch.no_grad():
    for data in tqdm(dl, desc='distill', leave=False):
        fronts = [data['fronts'][i].to(device, dtype=torch.float32)
                  for i in range(config.seq_len)]
        lidars = [data['lidars'][i].to(device, dtype=torch.float32)
                  for i in range(config.seq_len)]
        radars = [data['radars'][i].to(device, dtype=torch.float32)
                  for i in range(config.seq_len)]
        gps = data['gps'].to(device, dtype=torch.float32)
        logit_list.append(m.predict(fronts, lidars, radars, gps).cpu().numpy())
        gt_list.append(data['beamidx'][0].numpy())

del m; torch.cuda.empty_cache()
logits = np.concatenate(logit_list)
gt = np.concatenate(gt_list)
preds = np.argmax(logits, axis=1)
errors = np.abs(preds - gt)

# ── Figure 1: Error distribution by beam bin ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

bin_defs = [(0, 21, 'Low (0-21)', '#2196F3'),
            (22, 42, 'Mid (22-42)', '#FF9800'),
            (43, 63, 'High (43-63)', '#F44336')]

max_err = int(errors.max()) + 1
x = np.arange(0, min(max_err, 32))
width = 0.25

for i, (lo, hi, label, color) in enumerate(bin_defs):
    mask = (gt >= lo) & (gt <= hi)
    err_bin = errors[mask]
    counts = np.array([np.sum(err_bin == e) for e in x])
    ax.bar(x + i * width, counts / len(err_bin) * 100, width, label=label, color=color, alpha=0.8)

ax.set_xlabel('Prediction Error |pred beam − true beam|', fontsize=11)
ax.set_ylabel('Percentage of Samples (%)', fontsize=11)
ax.set_title('Prediction Error Distribution by Beam Range\n(Distilled Model)', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-0.5, 20)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/failure_error_dist.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: figures/failure_error_dist.pdf')

# ── Figure 2: Confusion matrix (beam prediction) ──────────────────────────────
# Group into 8 bins of 8 beams each for readability
n_bins = 8
bin_size = 64 // n_bins
conf = np.zeros((n_bins, n_bins))
for t, p in zip(gt, preds):
    conf[t // bin_size, p // bin_size] += 1

# Normalize by row
row_sums = conf.sum(axis=1, keepdims=True)
conf_norm = conf / (row_sums + 1e-8)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(conf_norm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Fraction of True Class Predicted')

tick_labels = [f'{i*bin_size}-{(i+1)*bin_size-1}' for i in range(n_bins)]
ax.set_xticks(range(n_bins))
ax.set_yticks(range(n_bins))
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(tick_labels, fontsize=9)
ax.set_xlabel('Predicted Beam Group', fontsize=11)
ax.set_ylabel('True Beam Group', fontsize=11)
ax.set_title('Beam Prediction Confusion Matrix (Grouped, Distilled Model)', fontsize=11)

for i in range(n_bins):
    for j in range(n_bins):
        ax.text(j, i, f'{conf_norm[i,j]:.2f}', ha='center', va='center',
                fontsize=8, color='black' if conf_norm[i,j] < 0.6 else 'white')

plt.tight_layout()
plt.savefig('figures/failure_confusion.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: figures/failure_confusion.pdf')

# ── Figure 3: Per-beam top-1 accuracy ────────────────────────────────────────
per_beam_acc = []
per_beam_n = []
for b in range(64):
    mask = gt == b
    n = mask.sum()
    if n > 0:
        acc = (preds[mask] == b).mean() * 100
    else:
        acc = float('nan')
    per_beam_acc.append(acc)
    per_beam_n.append(n)

per_beam_acc = np.array(per_beam_acc)
per_beam_n = np.array(per_beam_n)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top: per-beam accuracy
ax1.fill_between(range(64), per_beam_acc, alpha=0.4, color='#2196F3')
ax1.plot(range(64), per_beam_acc, color='#1565C0', linewidth=1.5)
ax1.axhline(y=np.nanmean(per_beam_acc), color='#F44336', linestyle='--',
            label=f'Mean {np.nanmean(per_beam_acc):.1f}%')
ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=11)
ax1.set_title('Per-Beam Top-1 Accuracy (Distilled Model)', fontsize=12)
ax1.legend()
ax1.set_ylim(0, 110)
ax1.grid(alpha=0.3)

# Shade beam bins
for lo, hi, color in [(0, 22, '#2196F3'), (22, 43, '#FF9800'), (43, 64, '#F44336')]:
    ax1.axvspan(lo, hi, alpha=0.05, color=color)

# Bottom: sample count per beam
bar_colors = ['#2196F3' if b <= 21 else '#FF9800' if b <= 42 else '#F44336'
              for b in range(64)]
ax2.bar(range(64), per_beam_n, color=bar_colors, alpha=0.7, width=0.8)
ax2.set_ylabel('# Test Samples', fontsize=11)
ax2.set_xlabel('Beam Index', fontsize=11)
ax2.set_title('Test Sample Distribution per Beam', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

# Legend for bin colors
from matplotlib.patches import Patch
legend_patches = [Patch(color='#2196F3', alpha=0.7, label='Low (0-21)'),
                  Patch(color='#FF9800', alpha=0.7, label='Mid (22-42)'),
                  Patch(color='#F44336', alpha=0.7, label='High (43-63)')]
ax2.legend(handles=legend_patches, fontsize=10)

plt.tight_layout()
plt.savefig('figures/failure_per_beam.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: figures/failure_per_beam.pdf')
print('\nAll failure analysis figures saved.')
