"""
Deployment metrics: inference latency, GPU memory, throughput, FLOPs.
Compares baseline TransFuser vs TransFuserV5 (v5b).
"""
import json, sys, time, numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '.')
from config_seq import GlobalConfig
from model_v5 import TransFuserV5
from model2_seq import TransFuser
from data2_seq import CARLA_Data

device = torch.device('cuda:0')

# ── Configs ───────────────────────────────────────────────────────────────────
config_v5 = GlobalConfig()
config_v5.add_velocity = 1; config_v5.add_mask = 0; config_v5.enhanced = 1
config_v5.angle_norm = 1; config_v5.custom_FoV_lidar = 1
config_v5.filtered = 0; config_v5.add_seg = 0; config_v5.n_layer = 2
config_v5.embd_pdrop = 0.3; config_v5.resid_pdrop = 0.3; config_v5.attn_pdrop = 0.3

config_bl = GlobalConfig()
config_bl.add_velocity = 1; config_bl.add_mask = 0; config_bl.enhanced = 1
config_bl.angle_norm = 1; config_bl.custom_FoV_lidar = 1
config_bl.filtered = 0; config_bl.add_seg = 0

# ── Load test data ────────────────────────────────────────────────────────────
data_root = config_v5.data_root + '/Multi_Modal/'
test_set = CARLA_Data(root=data_root, root_csv='scenario32_test_seq.csv',
                      config=config_v5, test=False)
dl_test = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

# ── Count parameters ──────────────────────────────────────────────────────────
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark_model(model, dl, n_warmup=10, n_bench=100,
                    use_forward=False, desc='model'):
    """Measure latency, memory, throughput."""
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    latencies = []
    sample_count = 0

    with torch.no_grad():
        for i, data in enumerate(dl):
            fronts, lidars, radars = [], [], []
            gps = data['gps'].to(device, dtype=torch.float32)
            for j in range(config_v5.seq_len):
                fronts.append(data['fronts'][j].to(device, dtype=torch.float32))
                lidars.append(data['lidars'][j].to(device, dtype=torch.float32))
                radars.append(data['radars'][j].to(device, dtype=torch.float32))

            # Warmup
            if i < n_warmup:
                if use_forward:
                    _ = model(fronts, lidars, radars, gps)
                else:
                    _ = model.predict(fronts, lidars, radars, gps)
                torch.cuda.synchronize(device)
                continue

            # Benchmark
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            if use_forward:
                _ = model(fronts, lidars, radars, gps)
            else:
                _ = model.predict(fronts, lidars, radars, gps)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)  # ms
            sample_count += 1

            if sample_count >= n_bench:
                break

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    lat = np.array(latencies)
    result = {
        'mean_latency_ms': float(np.mean(lat)),
        'p50_ms': float(np.percentile(lat, 50)),
        'p95_ms': float(np.percentile(lat, 95)),
        'p99_ms': float(np.percentile(lat, 99)),
        'throughput_fps': float(1000.0 / np.mean(lat)),
        'peak_gpu_mem_mb': float(peak_mem_mb),
        'n_samples': sample_count,
    }
    print(f'  [{desc}] latency: {result["mean_latency_ms"]:.2f}ms (p95={result["p95_ms"]:.2f}ms) '
          f'| throughput: {result["throughput_fps"]:.1f} fps '
          f'| peak mem: {result["peak_gpu_mem_mb"]:.0f} MB')
    return result


# ── Load models ───────────────────────────────────────────────────────────────
print('Loading models...')
model_v5b = TransFuserV5(config_v5, device).to(device)
model_v5b.load_state_dict(torch.load('log/s32_v5b/best_model.pth',
                                      map_location=device, weights_only=False))

model_bl = TransFuser(config_bl, device).to(device)
model_bl.load_state_dict(torch.load('log/s32_run3/best_model_baseline_backup.pth',
                                     map_location=device, weights_only=False))

# ── Count params ──────────────────────────────────────────────────────────────
v5b_total, v5b_trainable = count_params(model_v5b)
bl_total, bl_trainable = count_params(model_bl)
print(f'\n=== Parameter Count ===')
print(f'  Baseline:   total={bl_total/1e6:.1f}M, trainable={bl_trainable/1e6:.1f}M')
print(f'  TransFuserV5: total={v5b_total/1e6:.1f}M, trainable={v5b_trainable/1e6:.1f}M')
print(f'  Trainable reduction: {bl_trainable/v5b_trainable:.1f}x')

# ── FLOPs estimate ────────────────────────────────────────────────────────────
try:
    from thop import profile
    data_sample = next(iter(dl_test))
    fronts = [data_sample['fronts'][i].to(device, dtype=torch.float32)
              for i in range(config_v5.seq_len)]
    lidars = [data_sample['lidars'][i].to(device, dtype=torch.float32)
              for i in range(config_v5.seq_len)]
    radars = [data_sample['radars'][i].to(device, dtype=torch.float32)
              for i in range(config_v5.seq_len)]
    gps = data_sample['gps'].to(device, dtype=torch.float32)
    flops_v5b, _ = profile(model_v5b, inputs=(fronts, lidars, radars, gps),
                             verbose=False)
    print(f'\n  TransFuserV5 FLOPs: {flops_v5b/1e9:.2f} GFLOPs')
    flops_available = True
except Exception as e:
    print(f'\n  FLOPs: thop not available ({e}), skipping')
    flops_v5b = None
    flops_available = False

# ── Benchmark ─────────────────────────────────────────────────────────────────
print('\n=== Inference Benchmark (batch_size=1, GPU) ===')
results_v5b = benchmark_model(model_v5b, dl_test, desc='TransFuserV5')
del model_v5b; torch.cuda.empty_cache()

results_bl = benchmark_model(model_bl, dl_test, use_forward=True, desc='Baseline')
del model_bl; torch.cuda.empty_cache()

# ── Ensemble cost ──────────────────────────────────────────────────────────────
print(f'\n=== Ensemble Overhead ===')
ens_latency = results_v5b['mean_latency_ms'] * 4  # 4 models
print(f'  4-model ensemble estimated latency: {ens_latency:.2f}ms')
print(f'  Baseline: {results_bl["mean_latency_ms"]:.2f}ms')
print(f'  Ensemble overhead: {ens_latency/results_bl["mean_latency_ms"]:.1f}x')

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'param_count': {
        'baseline_total_M': bl_total / 1e6,
        'baseline_trainable_M': bl_trainable / 1e6,
        'v5b_total_M': v5b_total / 1e6,
        'v5b_trainable_M': v5b_trainable / 1e6,
        'trainable_reduction': bl_trainable / v5b_trainable,
    },
    'inference': {
        'v5b': results_v5b,
        'baseline': results_bl,
    },
    'ensemble_4model_latency_ms': ens_latency,
}
if flops_available and flops_v5b:
    out['flops'] = {'v5b_GFLOPs': flops_v5b / 1e9}

with open('log/deployment_metrics.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nSaved to log/deployment_metrics.json')
