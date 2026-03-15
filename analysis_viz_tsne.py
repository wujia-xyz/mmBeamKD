"""
t-SNE feature space visualization.
Extracts fusion token features and visualizes the learned beam representation.
Generates figures/tsne_v5b.pdf and figures/tsne_distill.pdf
"""
import sys, numpy as np, torch, torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE

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


def extract_features(model, dl, desc):
    """Extract the beam head input features (just before final FC)."""
    features_list, logits_list, gt_list = [], [], []
    hook_out = {}

    def hook_fn(module, input, output):
        hook_out['feat'] = output.detach().cpu()

    # Hook on join[3] (128-dim layer before final 64-class linear)
    # join = Linear(512,256) -> ReLU -> Dropout -> Linear(256,128) -> ReLU -> Dropout -> Linear(128,64)
    handle = model.join[4].register_forward_hook(
        lambda m, i, o: hook_out.update({'feat': o.detach().cpu()})
    )

    model.eval()
    with torch.no_grad():
        for data in tqdm(dl, desc=desc, leave=False):
            fronts = [data['fronts'][i].to(device, dtype=torch.float32)
                      for i in range(config.seq_len)]
            lidars = [data['lidars'][i].to(device, dtype=torch.float32)
                      for i in range(config.seq_len)]
            radars = [data['radars'][i].to(device, dtype=torch.float32)
                      for i in range(config.seq_len)]
            gps = data['gps'].to(device, dtype=torch.float32)
            logits = model.predict(fronts, lidars, radars, gps)
            features_list.append(hook_out['feat'].numpy())
            logits_list.append(logits.cpu().numpy())
            gt_list.append(data['beamidx'][0].numpy())

    handle.remove()
    feats = np.concatenate(features_list, 0)
    logits = np.concatenate(logits_list, 0)
    gt = np.concatenate(gt_list)
    preds = np.argmax(logits, axis=1)
    return feats, gt, preds


def plot_tsne(feats, gt, preds, title, save_path):
    print(f'  Running t-SNE for {title}...')
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    emb = tsne.fit_transform(feats)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # --- Plot 1: colored by true beam index ---
    cmap = plt.cm.get_cmap('plasma', 64)
    sc = axes[0].scatter(emb[:, 0], emb[:, 1], c=gt, cmap='plasma',
                         vmin=0, vmax=63, s=12, alpha=0.7, linewidths=0)
    plt.colorbar(sc, ax=axes[0], label='True Beam Index')
    axes[0].set_title('Colored by True Beam Index')
    axes[0].set_xlabel('t-SNE dim 1')
    axes[0].set_ylabel('t-SNE dim 2')
    axes[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # --- Plot 2: colored by prediction correctness ---
    errors = np.abs(preds - gt)
    correct = errors == 0
    near_correct = (errors > 0) & (errors <= 3)
    wrong = errors > 3

    axes[1].scatter(emb[wrong, 0], emb[wrong, 1], c='#e74c3c',
                    s=12, alpha=0.6, linewidths=0, label=f'Error>3 (n={wrong.sum()})')
    axes[1].scatter(emb[near_correct, 0], emb[near_correct, 1], c='#f39c12',
                    s=12, alpha=0.7, linewidths=0, label=f'Error 1-3 (n={near_correct.sum()})')
    axes[1].scatter(emb[correct, 0], emb[correct, 1], c='#2ecc71',
                    s=14, alpha=0.8, linewidths=0, label=f'Correct (n={correct.sum()})')
    axes[1].set_title('Colored by Prediction Accuracy')
    axes[1].set_xlabel('t-SNE dim 1')
    axes[1].legend(fontsize=9, markerscale=1.5)
    axes[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


# ── v5b model ─────────────────────────────────────────────────────────────────
print('\n[1/2] v5b model')
m = TransFuserV5(config, device).to(device)
m.load_state_dict(torch.load('log/s32_v5b/best_model.pth',
                              map_location=device, weights_only=False))
feats_v5b, gt, preds_v5b = extract_features(m, dl, 'v5b')
del m; torch.cuda.empty_cache()
plot_tsne(feats_v5b, gt, preds_v5b, 'TransFuserV5 (DBA=0.8058)', 'figures/tsne_v5b.pdf')

# ── distilled model ───────────────────────────────────────────────────────────
print('\n[2/2] Distilled model')
m = TransFuserV5(config, device).to(device)
m.load_state_dict(torch.load('log/s32_v10_distill/best_model.pth',
                              map_location=device, weights_only=False))
feats_distill, _, preds_distill = extract_features(m, dl, 'distill')
del m; torch.cuda.empty_cache()
plot_tsne(feats_distill, gt, preds_distill, 'Distilled Model (DBA=0.8285)', 'figures/tsne_distill.pdf')

print('\nDone. Figures saved to figures/')
