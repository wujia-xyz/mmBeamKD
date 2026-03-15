# mmBeamKD: Parameter-Efficient Multimodal Beam Prediction via Cross-Modal Knowledge Distillation

> **Paper**: *Parameter-Efficient Multimodal Beam Prediction for Vehicular mmWave Communications: A Scenario-32 Case Study with Knowledge Distillation*
>
> Xin Xie, Jia Wu*

---

## Overview

We propose **TransFuserV5**, a parameter-efficient multimodal transformer for mmWave beam prediction on the [DeepSense 6G](https://deepsense6g.net/) Scenario-32 benchmark (64-beam classification, 1905 training samples). Key contributions:

1. **Frozen-backbone fusion**: ResNet encoders frozen, only 21.7 M parameters trained (vs 78.4 M baseline), 1.66× faster inference.
2. **Ensemble knowledge distillation**: TransFuserV5-KD achieves **DBA = 0.8285** (+0.021 vs baseline, *p* = 0.0002).
3. **Cross-modal KD**: A camera+GPS-only student distilled from 4-modality teachers achieves **DBA = 0.8356** (+0.028, *p* < 0.0001) at 19.9 ms — matching the full 3-model ensemble with only 2 sensors.
4. **Sensor attribution**: Camera and GPS dominate in Scenario-32; LiDAR/radar are redundant.

## Main Results

| Method | Trainable Params | DBA | Top-1 (%) | Latency |
|--------|-----------------|-----|-----------|---------|
| TransFuser (baseline) | 78.4 M | 0.8076 | 32.60 | 33.1 ms |
| TransFuserV5 | 21.7 M | 0.8058 | 37.64 | 19.9 ms |
| **TransFuserV5-KD** | 21.7 M | **0.8285** | 35.91 | 19.9 ms |
| KD + Ensemble | — | 0.8353 | 38.11 | 80 ms |
| **CamGPS-only-KD** | 21.7 M | **0.8356** | 39.53 | 19.9 ms |

All comparisons use paired bootstrap significance testing (5000 resamples).

## Requirements

```bash
conda env create -f environment.yml
conda activate deepsense
```

## Dataset

Download [DeepSense 6G Scenario-32](https://deepsense6g.net/scenario_32/) and place under `Dataset/`. Update `data_root` in `config_seq.py`.

## Training

**Baseline:**
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_s32.py --id s32_baseline
```

**TransFuserV5:**
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_s32_v5.py --id s32_v5b
```

**TransFuserV5-KD (ensemble distillation):**
```bash
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29603 \
  python train_distill.py --id s32_kd --epochs 100
```

**CamGPS-only-KD (cross-modal distillation):**
```bash
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29604 \
  python train_distill_camgps.py --id s32_camgps_kd --epochs 60
```

## Evaluation & Analysis

```bash
python eval_ensemble_corrected.py      # Ensemble evaluation
python analysis_beam_bin.py            # Per-beam-bin DBA + modality ablation
python analysis_bootstrap_ci.py        # Bootstrap confidence intervals
python analysis_val_test_shift.py      # Distribution shift analysis
```

## Paper

The full paper draft is in `paper/main.pdf`. Figures can be regenerated with `paper/figures/gen_all_figs.py`.

## Acknowledgment

This work was supported in part by the Sichuan Natural Science Foundation under Grant 2026NSFSC0580.

---
*Corresponding author: Jia Wu (wujiahj@126.com)*
