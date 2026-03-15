"""
IEEE TWC publication style for matplotlib figures.

Rules:
- Font: 9pt serif (matches 10pt body text when scaled to column width)
- No titles inside figures (caption is in LaTeX \caption{})
- Single column: 3.5 inch wide  (88mm)
- Double column: 7.16 inch wide (182mm)
- Spines: keep left + bottom only
- Grid: y-only, light gray, no x-grid
- DPI: 300 for raster elements, save as PDF (vector)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── IEEE style ─────────────────────────────────────────────────────────────────
IEEE_RC = {
    'font.size':         9,
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize':    9,
    'axes.titlesize':    9,     # unused — no titles inside figures
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'legend.frameon':    True,
    'legend.framealpha': 0.9,
    'legend.edgecolor':  '#cccccc',
    'legend.handlelength': 1.5,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
    'lines.linewidth':   1.5,
    'lines.markersize':  5,
    'mathtext.fontset':  'stix',
    'text.usetex':       False,
}
plt.rcParams.update(IEEE_RC)

# Column widths
COL1 = 3.5   # single column (inches)
COL2 = 7.16  # double column (inches)

# Color palette (colorblind-safe tab10 subset)
COLORS = {
    'baseline': '#666666',
    'v5b':      '#2196F3',
    'distill':  '#F44336',
    'ensemble': '#9C27B0',
    'v9':       '#4CAF50',
    'low':      '#2196F3',
    'mid':      '#FF9800',
    'high':     '#F44336',
    'val':      '#2196F3',
    'test':     '#F44336',
}

def save(fig, name, outdir='/home/zqq/usb_disk/wujia/xindalao/DeepSense6G_TII-main/paper/figures'):
    """Save figure as PDF."""
    path = f'{outdir}/{name}.pdf'
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f'Saved: {path}')
    plt.close(fig)
