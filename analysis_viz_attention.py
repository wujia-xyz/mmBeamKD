"""
Modality contribution: encoder feature norm analysis (IEEE TWC style).
Regenerates:
  paper/figures/attention_modality_norms.pdf  -- overall mean activation bar chart
  paper/figures/attention_by_beam_bin.pdf     -- activation grouped by beam bin
No ax.set_title() calls. 9pt Times serif. Single-column width.
"""
import sys, os, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main')
sys.path.insert(0, '/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main/paper/figures')
from config_seq import GlobalConfig
from model_v5 import TransFuserV5
from data2_seq import CARLA_Data
from ieee_style import IEEE_RC, COL1, COLORS, save

plt.rcParams.update(IEEE_RC)

OUTDIR = '/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main/paper/figures'
CACHE  = '/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main/log/attention_norms_cache.npz'

# ── Load or compute per-sample activation norms ───────────────────────────────
if os.path.exists(CACHE):
    print('Loading cached activation data...')
    d = np.load(CACHE)
    cam_n, lid_n, rad_n, gps_n, gt = d['cam'], d['lid'], d['rad'], d['gps'], d['gt']
else:
    print('Computing per-sample feature norms (takes ~2 min)...')
    config = GlobalConfig()
    config.add_velocity = 1; config.add_mask = 0; config.enhanced = 1
    config.angle_norm = 1; config.custom_FoV_lidar = 1
    config.filtered = 0; config.add_seg = 0; config.n_layer = 2
    config.embd_pdrop = 0.3; config.resid_pdrop = 0.3; config.attn_pdrop = 0.3

    device = torch.device('cuda:0')
    data_root = config.data_root + '/Multi_Modal/'
    test_set = CARLA_Data(root=data_root, root_csv='scenario32_test_seq.csv',
                          config=config, test=False)
    dl = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    m = TransFuserV5(config, device).to(device)
    m.load_state_dict(torch.load(
        '/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main/log/s32_v5b/best_model.pth',
        map_location=device, weights_only=False))
    m.eval()

    def run_with_hooks(model, fronts, lidars, radars, gps):
        norms = {}
        hooks = []
        def make_hook(key):
            def hook(mod, inp, out):
                norms[key] = out.detach().abs().mean().item()
            return hook
        hooks.append(model.encoder.image_encoder.features.register_forward_hook(make_hook('cam')))
        hooks.append(model.encoder.lidar_encoder._model.layer4.register_forward_hook(make_hook('lid')))
        hooks.append(model.encoder.radar_encoder._model.layer4.register_forward_hook(make_hook('rad')))
        hooks.append(model.encoder.vel_emb1.register_forward_hook(make_hook('gps')))
        with torch.no_grad():
            logits = model.predict(fronts, lidars, radars, gps)
        for h in hooks:
            h.remove()
        return logits, norms

    raw = {'cam': [], 'lid': [], 'rad': [], 'gps': [], 'gt': []}
    with torch.no_grad():
        for data in tqdm(dl):
            fronts = [data['fronts'][j].to(device, dtype=torch.float32)
                      for j in range(config.seq_len)]
            lidars = [data['lidars'][j].to(device, dtype=torch.float32)
                      for j in range(config.seq_len)]
            radars = [data['radars'][j].to(device, dtype=torch.float32)
                      for j in range(config.seq_len)]
            gps = data['gps'].to(device, dtype=torch.float32)
            _, norms = run_with_hooks(m, fronts, lidars, radars, gps)
            raw['cam'].append(norms.get('cam', 0))
            raw['lid'].append(norms.get('lid', 0))
            raw['rad'].append(norms.get('rad', 0))
            raw['gps'].append(norms.get('gps', 0))
            raw['gt'].append(data['beamidx'][0].item())

    del m; torch.cuda.empty_cache()
    cam_n = np.array(raw['cam']); lid_n = np.array(raw['lid'])
    rad_n = np.array(raw['rad']); gps_n = np.array(raw['gps'])
    gt    = np.array(raw['gt'])
    np.savez(CACHE, cam=cam_n, lid=lid_n, rad=rad_n, gps=gps_n, gt=gt)
    print(f'Cached to {CACHE}')

# ── Shared setup ──────────────────────────────────────────────────────────────
modalities  = ['Camera', 'LiDAR', 'Radar', 'GPS']
all_norms   = [cam_n, lid_n, rad_n, gps_n]
colors_m    = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

# ── Figure A: Overall mean activation (bar + error bars) ─────────────────────
means = np.array([n.mean() for n in all_norms])
stds  = np.array([n.std()  for n in all_norms])

fig, ax = plt.subplots(figsize=(COL1, 2.4))
bars = ax.bar(modalities, means, yerr=stds, capsize=5,
              color=colors_m, alpha=0.85, width=0.55,
              edgecolor='white', linewidth=0.8,
              error_kw=dict(elinewidth=0.8, capthick=0.8, ecolor='#555'))
ax.set_ylabel('Mean activation')
ax.set_ylim(0, means.max() + stds.max() * 1.7)
ax.grid(axis='y', alpha=0.25, linewidth=0.6)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + stds.max() * 0.12,
            f'{val:.3f}', ha='center', fontsize=7.5, fontweight='bold')
save(fig, 'attention_modality_norms', outdir=OUTDIR)

# ── Figure B: Activation by beam bin (grouped bar chart) ─────────────────────
beam_bins = [('Low\n(0\u201321)',  0, 21),
             ('Mid\n(22\u201342)', 22, 42),
             ('High\n(43\u201363)',43, 63)]
x     = np.arange(len(beam_bins))
width = 0.20

fig, ax = plt.subplots(figsize=(COL1, 2.4))
for i, (norms_arr, mname, mcolor) in enumerate(zip(all_norms, modalities, colors_m)):
    vals_m = [norms_arr[(gt >= lo) & (gt <= hi)].mean() for _, lo, hi in beam_bins]
    vals_s = [norms_arr[(gt >= lo) & (gt <= hi)].std()  for _, lo, hi in beam_bins]
    ax.bar(x + (i - 1.5) * width, vals_m, width,
           label=mname, color=mcolor, alpha=0.85,
           edgecolor='white', linewidth=0.8,
           yerr=vals_s, capsize=3,
           error_kw=dict(elinewidth=0.7, capthick=0.7, ecolor='#555'))

ax.set_xticks(x)
ax.set_xticklabels([b[0] for b in beam_bins])
ax.set_ylabel('Mean activation')
ax.legend(fontsize=7.5, ncol=2, loc='upper right',
          handlelength=1.2, handletextpad=0.4, columnspacing=0.8)
ax.grid(axis='y', alpha=0.25, linewidth=0.6)
save(fig, 'attention_by_beam_bin', outdir=OUTDIR)

print('Done. Saved attention_modality_norms.pdf and attention_by_beam_bin.pdf')
