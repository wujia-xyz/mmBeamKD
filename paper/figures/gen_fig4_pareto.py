"""Generate Fig 4: DBA vs Latency Pareto plot."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'mathtext.fontset': 'stix',
})

# ── Data ──────────────────────────────────────────────────────────────────────
# (name, DBA, latency_ms, trainable_params_M, color, marker, ckpt_MB)
models = [
    ('Baseline\n(TransFuser)',   0.8076, 33.1,  78.4, '#888888', 's',  299.7),
    ('TransFuserV5\n(seed 42)',  0.8058, 19.9,  21.7, '#2196F3', 'o',  213.5),
    ('TransFuserV5\n(seed 123)', 0.8013, 19.9,  21.7, '#64B5F6', 'o',  213.5),
    ('TransFuserV5\n(seed 7)',   0.7973, 19.9,  21.7, '#90CAF9', 'o',  213.5),
    ('v9 (frozen BN)',           0.8066, 19.9,  21.7, '#4CAF50', '^',  213.5),
    ('TransFuserV5-KD\n(Ours)', 0.8285, 19.9,  21.7, '#F44336', 'D',  213.5),
    ('3-seed ensemble',          0.8217, 60.0,  21.7, '#FF9800', 'p',  640.5),
    ('distill+v5b+v9\nensemble\n(Ours)', 0.8353, 79.7, 21.7, '#9C27B0', '*', 640.5),
]

fig, ax = plt.subplots(figsize=(6, 4))

# Plot each model — size proportional to checkpoint size (sqrt for area)
for name, dba, lat, params, color, marker, ckpt in models:
    size = (ckpt / 100) ** 1.5 * 60  # scale marker size to checkpoint
    ax.scatter(lat, dba, s=size, c=color, marker=marker, zorder=5,
               edgecolors='white' if color not in ['#888888'] else 'black',
               linewidths=0.8, alpha=0.9)

# Legend entries (manual, for control)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#888888',
           markersize=9, label='Baseline (78.4M params)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
           markersize=9, label='TransFuserV5 (21.7M params)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#4CAF50',
           markersize=9, label='v9 frozen-BN fix'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#F44336',
           markersize=10, label='TransFuserV5-KD (ours)'),
    Line2D([0], [0], marker='p', color='w', markerfacecolor='#FF9800',
           markersize=10, label='3-seed ensemble'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#9C27B0',
           markersize=13, label='distill+v5b+v9 (ours)'),
]
ax.legend(handles=legend_handles, loc='lower right', frameon=True,
          framealpha=0.9, edgecolor='#cccccc', ncol=1)

# Annotations for key models
annotations = {
    'Baseline\n(TransFuser)':   (33.1, 0.8076, (38, 0.800), 'right'),
    'TransFuserV5-KD\n(Ours)': (19.9, 0.8285, (6, 0.832),  'left'),
    'distill+v5b+v9\nensemble\n(Ours)': (79.7, 0.8353, (62, 0.841), 'right'),
}
for name, dba, lat, params, color, marker, ckpt in models:
    key = name
    if key in annotations:
        _, _, (tx, ty), ha = annotations[key]
        ax.annotate(
            name.replace('\n', ' '),
            xy=(lat, dba), xytext=(tx, ty),
            fontsize=8, color=color,
            arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
            ha=ha,
        )

# Shading for Pareto-optimal front (approximate)
ax.axhline(0.8076, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5,
           label='Baseline DBA')

ax.set_xlabel('Inference Latency (ms, batch size = 1, RTX 4090)')
ax.set_ylabel('Test DBA')
ax.set_xlim(0, 100)
ax.set_ylim(0.780, 0.855)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
ax.grid(axis='y', alpha=0.25, linewidth=0.7)

# Marker size legend (checkpoint size)
for ckpt_mb, label in [(213.5, '213 MB'), (299.7, '300 MB'), (640.5, '640 MB')]:
    size = (ckpt_mb / 100) ** 1.5 * 60
    ax.scatter([], [], s=size, c='gray', alpha=0.5, label=f'Ckpt: {label}')
ax.legend(handles=legend_handles, loc='lower right', frameon=True,
          framealpha=0.9, edgecolor='#cccccc')

fig.savefig('fig4_pareto.pdf', dpi=300, bbox_inches='tight')
print('Saved: fig4_pareto.pdf')
