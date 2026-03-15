"""
Generate all data-driven paper figures in IEEE TWC style.
No internal titles. Fonts >= 9pt. Single/double column widths.
Run from: DeepSense6G_TII-main/paper/figures/
"""
import sys, json, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.special import softmax
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

sys.path.insert(0, '/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main')
from config_seq import GlobalConfig
from model_v5 import TransFuserV5
from model2_seq import TransFuser
from data2_seq import CARLA_Data

sys.path.insert(0, '/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main/paper/figures')
from ieee_style import IEEE_RC, COL1, COL2, COLORS, save
plt.rcParams.update(IEEE_RC)

# ── Load data & models ────────────────────────────────────────────────────────
device = torch.device('cuda:0')
config = GlobalConfig()
config.add_velocity=1; config.add_mask=0; config.enhanced=1; config.angle_norm=1
config.custom_FoV_lidar=1; config.filtered=0; config.add_seg=0; config.n_layer=2
config.embd_pdrop=0.3; config.resid_pdrop=0.3; config.attn_pdrop=0.3
data_root = config.data_root + '/Multi_Modal/'

test_set = CARLA_Data(root=data_root, root_csv='scenario32_test_seq.csv',
                      config=config, test=False)
val_set  = CARLA_Data(root=data_root, root_csv='scenario32_val_seq.csv',
                      config=config, test=False)
dl_test = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)
dl_val  = DataLoader(val_set,  batch_size=8, shuffle=False, num_workers=4)

def get_logits(model, dl, use_forward=False):
    model.eval(); logs, gts = [], []
    with torch.no_grad():
        for data in dl:
            fronts=[data['fronts'][i].to(device,dtype=torch.float32) for i in range(config.seq_len)]
            lidars=[data['lidars'][i].to(device,dtype=torch.float32) for i in range(config.seq_len)]
            radars=[data['radars'][i].to(device,dtype=torch.float32) for i in range(config.seq_len)]
            gps=data['gps'].to(device,dtype=torch.float32)
            if use_forward: lgt=model(fronts,lidars,radars,gps)
            else: lgt=model.predict(fronts,lidars,radars,gps)
            logs.append(lgt.cpu().numpy()); gts.append(data['beamidx'][0].numpy())
    return np.concatenate(logs), np.concatenate(gts)

def get_feats(model, dl):
    """Extract 128-dim pre-classifier features via hook."""
    model.eval(); feats, logs, gts = [], [], []
    hook_out = {}
    handle = model.join[4].register_forward_hook(
        lambda m,i,o: hook_out.update({'f': o.detach().cpu()}))
    with torch.no_grad():
        for data in dl:
            fronts=[data['fronts'][i].to(device,dtype=torch.float32) for i in range(config.seq_len)]
            lidars=[data['lidars'][i].to(device,dtype=torch.float32) for i in range(config.seq_len)]
            radars=[data['radars'][i].to(device,dtype=torch.float32) for i in range(config.seq_len)]
            gps=data['gps'].to(device,dtype=torch.float32)
            lgt=model.predict(fronts,lidars,radars,gps)
            feats.append(hook_out['f'].numpy())
            logs.append(lgt.cpu().numpy()); gts.append(data['beamidx'][0].numpy())
    handle.remove()
    return np.concatenate(feats), np.concatenate(logs), np.concatenate(gts)

# Load models
print('Loading models...')
config_bl = GlobalConfig()
config_bl.add_velocity=1; config_bl.add_mask=0; config_bl.enhanced=1; config_bl.angle_norm=1
config_bl.custom_FoV_lidar=1; config_bl.filtered=0; config_bl.add_seg=0

m_bl = TransFuser(config_bl,device).to(device)
m_bl.load_state_dict(torch.load('/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main/log/s32_run3/best_model_baseline_backup.pth',
                                  map_location=device, weights_only=False))
lgt_bl, gt = get_logits(m_bl, dl_test, use_forward=True)
del m_bl; torch.cuda.empty_cache()

m_v5b = TransFuserV5(config,device).to(device)
m_v5b.load_state_dict(torch.load('/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main/log/s32_v5b/best_model.pth',
                                   map_location=device, weights_only=False))
lgt_v5b, _ = get_logits(m_v5b, dl_test)

m_kd = TransFuserV5(config,device).to(device)
m_kd.load_state_dict(torch.load('/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main/log/s32_v10_distill/best_model.pth',
                                  map_location=device, weights_only=False))
lgt_kd, _ = get_logits(m_kd, dl_test)

m_v9 = TransFuserV5(config,device).to(device)
m_v9.load_state_dict(torch.load('/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main/log/s32_v9/best_model.pth',
                                  map_location=device, weights_only=False))
lgt_v9, _ = get_logits(m_v9, dl_test)

# Ensemble
lgt_ens = np.mean([softmax(lgt_kd,axis=1), softmax(lgt_v5b,axis=1),
                   softmax(lgt_v9,axis=1)], axis=0)

pred_bl  = np.argsort(-lgt_bl,  axis=1)
pred_v5b = np.argsort(-lgt_v5b, axis=1)
pred_kd  = np.argsort(-lgt_kd,  axis=1)
pred_ens = np.argsort(-lgt_ens,  axis=1)
errors   = np.abs(pred_kd[:,0] - gt)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4: Accuracy–Latency Pareto Plot
# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Fig 4 (Pareto)...')
models_data = [
    ('Baseline',              0.8076, 33.1, 78.4, COLORS['baseline'], 's',  8.5),
    ('TransFuserV5 (s=42)',   0.8058, 19.9, 21.7, COLORS['v5b'],      'o',  7.5),
    ('TransFuserV5 (s=123)',  0.8013, 19.9, 21.7, '#64B5F6',           'o',  6.0),
    ('TransFuserV5 (s=7)',    0.7973, 19.9, 21.7, '#90CAF9',           'o',  6.0),
    ('v9 (frozen BN)',        0.8066, 19.9, 21.7, COLORS['v9'],        '^',  7.5),
    ('KD single (ours)',      0.8285, 19.9, 21.7, COLORS['distill'],   'D',  9.5),
    ('3-seed ensemble',       0.8217, 60.0, 21.7, COLORS['mid'],       'p',  8.0),
    ('KD+ens (ours)',         0.8353, 79.7, 21.7, COLORS['ensemble'],  '*',  12.0),
]

fig, ax = plt.subplots(figsize=(COL2*0.55, 2.5))

for name, dba, lat, _, color, marker, ms in models_data:
    ax.scatter(lat, dba, s=ms**2, c=color, marker=marker, zorder=5,
               edgecolors='white', linewidths=0.5, alpha=0.92)

ax.axhline(0.8076, color=COLORS['baseline'], linestyle='--',
           linewidth=0.9, alpha=0.6, label='Baseline DBA')

# Annotations for key points
annots = {
    'Baseline':         (33.1, 0.8076, (+3.5, -0.010), 'left'),
    'KD single (ours)': (19.9, 0.8285, (-3.5, +0.006), 'right'),
    'KD+ens (ours)':    (79.7, 0.8353, (-5.0, +0.004), 'right'),
}
for name, dba, lat, _, color, marker, ms in models_data:
    if name in annots:
        _, _, (dx, dy), ha = annots[name]
        ax.annotate(name, xy=(lat, dba), xytext=(lat+dx, dba+dy),
                    fontsize=7.5, color=color, ha=ha,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

legend_elems = [
    Line2D([0],[0],marker='s',color='w',markerfacecolor=COLORS['baseline'],markersize=8,label='Baseline (78.4M)'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=COLORS['v5b'],markersize=7,label='TransFuserV5 (21.7M)'),
    Line2D([0],[0],marker='^',color='w',markerfacecolor=COLORS['v9'],markersize=7,label='v9 frozen-BN fix'),
    Line2D([0],[0],marker='D',color='w',markerfacecolor=COLORS['distill'],markersize=8,label='KD single (ours)'),
    Line2D([0],[0],marker='p',color='w',markerfacecolor=COLORS['mid'],markersize=8,label='3-seed ensemble'),
    Line2D([0],[0],marker='*',color='w',markerfacecolor=COLORS['ensemble'],markersize=10,label='KD+ens (ours)'),
]
ax.legend(handles=legend_elems, loc='lower right', fontsize=7.5, ncol=1,
          framealpha=0.9, edgecolor='#ccc')
ax.set_xlabel('Inference Latency (ms)')
ax.set_ylabel('Test DBA')
ax.set_xlim(0, 98)
ax.set_ylim(0.785, 0.850)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.3f}'))
ax.grid(axis='y', alpha=0.2, linewidth=0.6)
save(fig, 'fig4_pareto', outdir='/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main/paper/figures')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5: t-SNE Feature Embeddings (v5b vs distilled) — double column
# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Fig 5 (t-SNE)...')
# Features
feats_v5b, _, _ = get_feats(m_v5b, dl_test)
del m_v5b; torch.cuda.empty_cache()
feats_kd, _, _  = get_feats(m_kd,  dl_test)
del m_kd;  torch.cuda.empty_cache()

print('  t-SNE v5b...')
emb_v5b = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(feats_v5b)
print('  t-SNE distill...')
emb_kd  = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(feats_kd)

errs_kd = np.abs(pred_kd[:,0] - gt)

fig, axes = plt.subplots(2, 2, figsize=(COL2, 3.6))

for row, (emb, preds, title_tag) in enumerate(
        [(emb_v5b, pred_v5b, 'TransFuserV5'),
         (emb_kd,  pred_kd,  'TransFuserV5-KD (ours)')]):
    errors_row = np.abs(preds[:,0] - gt)

    # Col 0: color by beam index
    sc = axes[row,0].scatter(emb[:,0], emb[:,1], c=gt, cmap='plasma',
                              vmin=0, vmax=63, s=4, alpha=0.7, linewidths=0)
    plt.colorbar(sc, ax=axes[row,0], label='Beam index', pad=0.02, shrink=0.9)
    axes[row,0].set_ylabel(title_tag, fontsize=8.5, fontweight='bold')

    # Col 1: color by prediction error
    wrong  = errors_row >  3
    near   = (errors_row > 0) & (errors_row <= 3)
    correct= errors_row == 0
    axes[row,1].scatter(emb[wrong,0],   emb[wrong,1],   c='#e74c3c', s=4, alpha=0.6, linewidths=0, label=f'Err>3 (n={wrong.sum()})')
    axes[row,1].scatter(emb[near,0],    emb[near,1],    c='#f39c12', s=4, alpha=0.7, linewidths=0, label=f'Err 1–3 (n={near.sum()})')
    axes[row,1].scatter(emb[correct,0], emb[correct,1], c='#2ecc71', s=5, alpha=0.8, linewidths=0, label=f'Correct (n={correct.sum()})')
    axes[row,1].legend(fontsize=7, markerscale=2.5, loc='upper right')

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

fig.tight_layout(pad=0.3)
save(fig, 'tsne_combined',
     outdir='/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main/paper/figures')
print('  Saved as tsne_combined.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6: Modality Feature Norms
# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Fig 6 (modality norms)...')
with open('/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main/log/beam_bin_analysis.json') as f:
    bb_data = json.load(f)

abl = bb_data['modality_ablation']
configs_abl = ['full','no_img','no_lid','no_rad','no_gps']
labels_abl  = ['All','No\nCamera','No\nLiDAR','No\nRadar','No\nGPS']
dba_vals    = [abl[c]['DBA'] for c in configs_abl]
colors_abl  = [COLORS['v5b'],'#e74c3c','#4CAF50','#FF9800','#9C27B0']

fig, ax = plt.subplots(figsize=(COL1, 2.4))
bars = ax.bar(labels_abl, dba_vals, color=colors_abl, alpha=0.85, width=0.6,
              edgecolor='white', linewidth=0.8)
ax.axhline(dba_vals[0], color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_ylabel('Test DBA')
ax.set_ylim(0.42, 0.85)
ax.grid(axis='y', alpha=0.25, linewidth=0.6)
for bar, val in zip(bars, dba_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{val:.3f}', ha='center', fontsize=7.5, fontweight='bold')
save(fig, 'attention_modality_norms')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7: Beam Distribution Shift
# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Fig 7 (distribution shift)...')

# GT beam indices for val
gt_val_list = []
for data in dl_val:
    gt_val_list.append(data['beamidx'][0].numpy())
gt_val = np.concatenate(gt_val_list)

fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6))

# (a) Histogram + KDE
bins = np.arange(0, 65, 2)
axes[0].hist(gt_val, bins=bins, density=True, alpha=0.55, color=COLORS['val'],
             label=f'Val (μ={gt_val.mean():.1f})')
axes[0].hist(gt,     bins=bins, density=True, alpha=0.55, color=COLORS['test'],
             label=f'Test (μ={gt.mean():.1f})')
x = np.linspace(0, 63, 300)
axes[0].plot(x, gaussian_kde(gt_val, bw_method=0.2)(x),
             color=COLORS['val'],  linewidth=1.8)
axes[0].plot(x, gaussian_kde(gt,     bw_method=0.2)(x),
             color=COLORS['test'], linewidth=1.8)
axes[0].axvline(gt_val.mean(), color=COLORS['val'],  linestyle='--', linewidth=1.0, alpha=0.8)
axes[0].axvline(gt.mean(),     color=COLORS['test'], linestyle='--', linewidth=1.0, alpha=0.8)
axes[0].set_xlabel('Beam index')
axes[0].set_ylabel('Density')
axes[0].legend(fontsize=8)
axes[0].grid(axis='y', alpha=0.2, linewidth=0.6)

# (b) CDF comparison
b_range = np.arange(0, 64)
val_cdf  = [(gt_val<=b).mean() for b in b_range]
test_cdf = [(gt    <=b).mean() for b in b_range]
axes[1].plot(b_range, val_cdf,  color=COLORS['val'],  linewidth=1.8, label='Val')
axes[1].plot(b_range, test_cdf, color=COLORS['test'], linewidth=1.8, label='Test')
axes[1].fill_between(b_range, val_cdf, test_cdf, alpha=0.15, color='gray')
ks_idx = np.argmax(np.abs(np.array(val_cdf)-np.array(test_cdf)))
axes[1].annotate(f'Max Δ={abs(val_cdf[ks_idx]-test_cdf[ks_idx]):.3f}\nat beam {ks_idx}',
                 xy=(ks_idx, (val_cdf[ks_idx]+test_cdf[ks_idx])/2),
                 xytext=(ks_idx+8, 0.3), fontsize=7.5,
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
axes[1].set_xlabel('Beam index')
axes[1].set_ylabel('Cumulative probability')
axes[1].legend(fontsize=8, loc='upper left')
axes[1].grid(axis='y', alpha=0.2, linewidth=0.6)

fig.tight_layout(pad=0.5)
save(fig, 'shift_beam_dist')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7b: Per-beam val-test accuracy gap
# ═══════════════════════════════════════════════════════════════════════════════
# Run val inference with distilled model
m_kd2 = TransFuserV5(config,device).to(device)
m_kd2.load_state_dict(torch.load('/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main/log/s32_v10_distill/best_model.pth',
                                   map_location=device, weights_only=False))
lgt_kd_val, gt_val2 = get_logits(m_kd2, dl_val)
del m_kd2; torch.cuda.empty_cache()
pred_kd_val = np.argmax(lgt_kd_val, axis=1)
pred_kd_test = pred_kd[:,0]

beams_present = []
val_accs, test_accs = [], []
for b in range(64):
    mv = (gt_val2==b); mt = (gt==b)
    if mv.sum()>0 and mt.sum()>0:
        beams_present.append(b)
        val_accs.append((pred_kd_val[mv]==b).mean()*100)
        test_accs.append((pred_kd_test[mt]==b).mean()*100)

val_accs = np.array(val_accs); test_accs = np.array(test_accs)
gap = val_accs - test_accs

fig, axes = plt.subplots(2, 1, figsize=(COL2*0.75, 3.5), sharex=True)

axes[0].plot(beams_present, val_accs,  color=COLORS['val'],  linewidth=1.5, label='Val')
axes[0].plot(beams_present, test_accs, color=COLORS['test'], linewidth=1.5, label='Test')
axes[0].set_ylabel('Top-1 Accuracy (%)')
axes[0].legend(fontsize=8)
axes[0].grid(axis='y', alpha=0.2, linewidth=0.6)

bar_colors = [COLORS['val'] if g>=0 else COLORS['test'] for g in gap]
axes[1].bar(beams_present, gap, color=bar_colors, alpha=0.75, width=0.9)
axes[1].axhline(0, color='black', linewidth=0.7)
axes[1].axhline(gap.mean(), color='orange', linestyle='--', linewidth=1.0,
                label=f'Mean gap {gap.mean():.1f}%')
axes[1].set_xlabel('Beam index')
axes[1].set_ylabel('Val − Test Acc. (%)')
axes[1].legend(fontsize=8)
axes[1].grid(axis='y', alpha=0.2, linewidth=0.6)

fig.tight_layout(pad=0.4)
save(fig, 'shift_acc_gap')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8: Failure Analysis (per-beam + error histogram + confusion)
# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Fig 8 (failure analysis)...')

# (a) Per-beam top-1 accuracy + sample count
per_beam_acc = []; per_beam_n = []
for b in range(64):
    mask = (gt==b); n = mask.sum()
    acc = (pred_kd[:,0][mask]==b).mean()*100 if n>0 else np.nan
    per_beam_acc.append(acc); per_beam_n.append(n)
per_beam_acc = np.array(per_beam_acc); per_beam_n = np.array(per_beam_n)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL2*0.85, 3.8), sharex=True)

x = np.arange(64)
bin_colors = [COLORS['low'] if b<=21 else COLORS['mid'] if b<=42 else COLORS['high']
              for b in range(64)]
ax1.fill_between(x, per_beam_acc, alpha=0.25, color='steelblue')
ax1.plot(x, per_beam_acc, color='steelblue', linewidth=1.4)
ax1.axhline(np.nanmean(per_beam_acc), color=COLORS['test'], linestyle='--',
            linewidth=0.9, label=f'Mean {np.nanmean(per_beam_acc):.1f}%')
# Shade bins
for lo, hi, c in [(0,22,COLORS['low']),(22,43,COLORS['mid']),(43,64,COLORS['high'])]:
    ax1.axvspan(lo, hi, alpha=0.06, color=c)
ax1.set_ylabel('Top-1 Accuracy (%)')
ax1.set_ylim(0, 105)
ax1.legend(fontsize=8)
ax1.grid(axis='y', alpha=0.2, linewidth=0.6)

ax2.bar(x, per_beam_n, color=bin_colors, alpha=0.78, width=0.9)
ax2.set_xlabel('Beam index')
ax2.set_ylabel('# Test samples')
legend_patches = [mpatches.Patch(color=COLORS['low'],alpha=0.78,label='Low (0–21)'),
                  mpatches.Patch(color=COLORS['mid'],alpha=0.78,label='Mid (22–42)'),
                  mpatches.Patch(color=COLORS['high'],alpha=0.78,label='High (43–63)')]
ax2.legend(handles=legend_patches, fontsize=8, loc='upper right')
ax2.grid(axis='y', alpha=0.2, linewidth=0.6)

fig.tight_layout(pad=0.4)
save(fig, 'failure_per_beam')

# (b) Error distribution by beam bin
fig, ax = plt.subplots(figsize=(COL1*0.9, 2.3))
x_err = np.arange(0, 20)
width = 0.28
for i, (lo, hi, label, color) in enumerate(
        [(0,21,'Low (0–21)',COLORS['low']),(22,42,'Mid (22–42)',COLORS['mid']),
         (43,63,'High (43–63)',COLORS['high'])]):
    mask = (gt>=lo)&(gt<=hi)
    errs = errors[mask]
    counts = np.array([np.sum(errs==e) for e in x_err]) / len(errs) * 100
    ax.bar(x_err+i*width, counts, width, label=label, color=color, alpha=0.82)

ax.set_xlabel('Prediction error |ŷ − y*|')
ax.set_ylabel('Fraction (%)')
ax.set_xlim(-0.4, 15)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.2, linewidth=0.6)
save(fig, 'failure_error_dist')

# (c) Confusion matrix (grouped 8 bins × 8 bins)
n_bins = 8; bin_size = 64//n_bins
conf = np.zeros((n_bins,n_bins))
for t,p in zip(gt, pred_kd[:,0]):
    conf[t//bin_size, p//bin_size] += 1
conf_norm = conf / (conf.sum(axis=1, keepdims=True) + 1e-8)

fig, ax = plt.subplots(figsize=(COL1, COL1*0.9))
im = ax.imshow(conf_norm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Fraction', shrink=0.9, pad=0.02)
tick_labels = [f'{i*bin_size}–{(i+1)*bin_size-1}' for i in range(n_bins)]
ax.set_xticks(range(n_bins)); ax.set_yticks(range(n_bins))
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(tick_labels, fontsize=7)
ax.set_xlabel('Predicted beam group')
ax.set_ylabel('True beam group')
for i in range(n_bins):
    for j in range(n_bins):
        ax.text(j,i,f'{conf_norm[i,j]:.2f}', ha='center', va='center',
                fontsize=6.5, color='black' if conf_norm[i,j]<0.65 else 'white')
save(fig, 'failure_confusion')

print('\nAll figures generated successfully.')
