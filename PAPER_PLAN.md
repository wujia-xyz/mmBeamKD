# Paper Plan — TWC Submission

**Title Option 1**: Parameter-Efficient Multimodal Beam Prediction for Vehicular mmWave Communications: A Scenario-32 Case Study
**Title Option 2**: Knowledge Distillation and Sensor Dominance Analysis for Multimodal Vehicular mmWave Beam Prediction Under Distribution Shift
**Venue**: IEEE Transactions on Wireless Communications (TWC)
**Type**: Method + Empirical + Analysis
**Target length**: ~12 pages (IEEE double-column, IEEEtran)
**Date**: 2026-03-15
**Reviewer score**: 8/10 (Weak Accept)

---

## Core Narrative

In DeepSense 6G Scenario-32, a parameter-efficient multimodal transformer with frozen modality backbones matches baseline DBA at much lower cost, while knowledge distillation and probability-space ensembling deliver significant DBA gains; ablations and diagnostics show camera+GPS dominance and remaining vulnerability to beam-distribution shift.

**Framing rules**:
- Do NOT claim single-model SOTA on DBA (v5b matches, not beats, baseline)
- Do NOT generalize camera+GPS finding beyond Scenario-32
- Do NOT claim distribution shift is solved

---

## Claims-Evidence Matrix

| # | Claim | Safe TWC wording | Evidence | Where |
|---|-------|-----------------|----------|-------|
| C1 | Efficiency | TransFuserV5 achieves baseline-level DBA with substantially lower training and deployment cost | v5b DBA 0.8058 vs 0.8076 (p=0.60); 21.7M vs 78.4M trainable; 19.9ms vs 33.1ms; 213.5MB vs 299.7MB | Abstract, Intro, Table II |
| C2 | Modality dominance | Camera and GPS provide most predictive signal in S32; LiDAR/radar add limited marginal benefit | No-camera DBA 0.503; no-GPS DBA 0.470; no-LiDAR DBA 0.807; CamGPS-only DBA 0.806 | Intro, Table III, Analysis |
| C3 | Distillation gain | KD significantly improves DBA over both baseline and undistilled student | DBA 0.8285, CI [0.807, 0.849], p=0.0002 | Abstract, Method, Experiments |
| C4 | Ensemble gain | Probability-space ensemble achieves best DBA at higher latency | DBA 0.8353, CI [0.815, 0.855], p<0.0001; ~80ms | Experiments, Discussion |
| C5 | Shift failure | Residual errors concentrated in high-beam bins, linked to val-test shift | Val mean 21.9 vs test 18.6; high-bin DBA 0.694 vs low-bin 0.819 | Intro, Analysis |
| C6 | Representation | t-SNE/attention qualitatively supports camera+GPS and distillation sharpening | tsne_*.pdf, attention_*.pdf | Analysis only |
| C7 | Stat rigor | Bootstrap CIs separate matched performance from significant gains | All p-values in Table II | Experiments |

---

## Section Structure

| § | Title | Length | Figures/Tables |
|---|-------|--------|---------------|
| 0 | Abstract | 0.25 pp | — |
| 1 | Introduction | 1.2 pp | — |
| 2 | Related Work | 0.9 pp | — |
| 3 | System Model & Problem Formulation | 1.1 pp | Fig. 1, Table I |
| 4 | Proposed Method | 2.3 pp | Fig. 2, Fig. 3 |
| 5 | Experiments | 2.8 pp | Table II, Table III, Fig. 4 |
| 6 | Analysis & Discussion | 2.8 pp | Table IV, Fig. 5-8 |
| 7 | Conclusion | 0.3 pp | — |
| **Total** | | **~11.7 pp** | |

---

## Figure Plan

| Fig. | Type | Content | Section |
|------|------|---------|---------|
| Fig. 1 | NEW — Task schematic | S32 setup: 5-frame cam/LiDAR/radar+GPS → 64-beam, DBA metric | §3 |
| Fig. 2 | NEW — Architecture | Frozen backbones + cross-attention fusion + GPT layers + head | §4 |
| Fig. 3 | NEW — Distillation | Teacher ensemble → student training pipeline | §4 |
| Fig. 4 | NEW — Pareto plot | DBA vs latency, marker=checkpoint size | §5 |
| Fig. 5 | Existing: tsne_v5b + tsne_distill | t-SNE feature space comparison | §6 |
| Fig. 6 | Existing: attention_modality_norms + attention_by_beam_bin | Modality contribution analysis | §6 |
| Fig. 7 | Existing: shift_beam_dist + shift_acc_gap | Distribution shift visualization | §6 |
| Fig. 8 | Existing: failure_per_beam + failure_confusion + failure_error_dist | Failure analysis | §6 |

### Tables

| Table | Content |
|-------|---------|
| Table I | Dataset split, modalities, target, DBA metric definition |
| Table II | Main results: all models, DBA ± CI, p-value, Top-1, latency, checkpoint |
| Table III | Modality ablation + CamGPS-only control |
| Table IV | Per-beam-bin performance (distilled model) |

---

## Citation Plan

~35-45 references, 65-70% from IEEE wireless venues.

- **mmWave/vehicular beam prediction** (largest block): DeepSense, TWC, TCom, TVT, JSAC papers
- **Sensing-aided/multimodal beam prediction**: camera/LiDAR/radar-assisted, ISAC
- **Transformer fusion** (short): TransFuser, multimodal sensor fusion
- **Knowledge distillation**: wireless/edge deployment first, then general ML
- **Distribution shift in wireless**: domain adaptation, robustness
- **Statistical evaluation**: bootstrap CI methodology (small block)

---

## Key Metrics Summary

```
Baseline DBA:        0.8076 [0.786, 0.829]
v5b DBA:             0.8058 [0.783, 0.829]  (p=0.60, not sig.)
Distilled DBA:       0.8285 [0.807, 0.849]  (p=0.0002, SIG)
Ensemble DBA:        0.8353 [0.815, 0.855]  (p<0.0001, SIG)
CamGPS-only DBA:     0.8064
No-camera DBA:       0.5028  (-37.5%)
No-GPS DBA:          0.4702  (-41.7%)
Speedup:             1.66×   (19.9ms vs 33.1ms)
Checkpoint saving:   29%     (213.5 vs 299.7 MB)
High-bin DBA:        0.694   (beams 43-63, n=49)
```

---

## Next Steps

- [x] Create paper/ folder structure
- [ ] Generate new figures (Fig. 1-4) using paper-figure skill
- [ ] Write LaTeX sections using paper-write skill
- [ ] Compile PDF using paper-compile skill
