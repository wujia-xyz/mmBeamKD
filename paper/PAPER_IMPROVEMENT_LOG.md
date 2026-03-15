# Paper Improvement Log

## Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Round 0 (original) | —/10 | — | First draft, 10 pages |
| Round 1 (prior session) | 6/10 | Almost | Physical DBA interpretation, modality causality, statistical protocol, KD transparency, train-test shift, softened claims |
| Round 2 (prior session) | 6/10 | Almost | Arithmetic fix (36%→90%), beam-budget table, deployment note softened |
| Round 3 (this session) | 5/10 | Almost | Fresh review of restructured paper (figures split, sections merged, IEEE fonts) |
| Round 4 (this session) | 6/10 | Almost | CamGPS-only in Table I, beam-probing overhead, beam coherence time, DBA-as-gain formalization, prior work comparison table, KD variance disclosure |

## Round 3 Review & Fixes (this session, Round 1)

<details>
<summary>GPT-5.4 xhigh Review (Round 3 / Session Round 1)</summary>

**Score**: 5/10 (borderline reject for TWC)

**CRITICAL:**
1. No communication-level evaluation beyond DBA/Top-1
2. Journal-level novelty not established; single-scenario validation

**MAJOR:**
1. CamGPS-only not fully characterized in main table (no CI, latency, checkpoint)
2. KD result from single student run — training variance not addressed
3. Deployment discussion: 19.9ms/80ms not mapped to beam-update budget

**MINOR:**
1. One-sided p-value usage needs clearer justification
2. Distribution-shift explanation correlational only
3. Relative positioning against prior DeepSense work too limited

**Verdict**: Almost

</details>

### Fixes Implemented (Round 3 → Round 4)
1. **[CRITICAL] DBA-as-beamforming-gain**: Added formal paragraph showing DBA = E[G_norm^(K)] under piecewise-linear gain model; 82.9% oracle gain for KD vs 80.8% baseline
2. **[CRITICAL] Beam-probing overhead**: Added E[probes] = B + (1-HitRate(B))×(B_total-B) formula; KD gives 15.9% fewer probes at B=5
3. **[MAJOR] CamGPS-only fully characterized**: Added to Table I with DBA=0.8064, CI=[0.784,0.828], p=0.566, 21.7M params, 19.9ms, 213.5MB (bootstrap computed from scratch)
4. **[MAJOR] KD variance note**: Added ±0.003 hyperparameter sensitivity; acknowledged single-run limitation with CI lower bound argument (+0.010 >> ±0.003)
5. **[MAJOR] Beam coherence time**: T_beam(r) = Δθ·r/v equation; at 50km/h, r=10m: 35ms > 19.9ms (single model OK); ensemble needs r≥23m
6. **Explicit case-study framing**: Added paragraph in intro stating all results are Scenario-32 specific
7. **Prior work comparison table**: Table showing frozen enc. / sensor attr. / statistical eval. vs TransFuser / DeepSense top-1

## Round 4 Review & Fixes (this session, Round 2)

<details>
<summary>GPT-5.4 xhigh Review (Round 4 / Session Round 2)</summary>

**Score**: 6/10

**No CRITICAL issues remain.**

**MAJOR:**
1. Novelty/generality — single scenario, incremental novelty (framing fix implemented)
2. CamGPS-only-KD experiment not reported
3. KD multi-seed variance partially addressed

**MINOR:**
1. Beamforming gain is modeled equivalence, not direct link-level metric
2. Probing overhead depends on one specific fallback protocol
3. Latency breakdown not specified

**Verdict**: Almost

</details>

### Fixes Implemented (Round 4)
1. **CamGPS-only-KD training launched**: Training `s32_v12_camgps_kd` in background (GPU 1). Epoch 0: Val=0.8421, Test=0.8079 (baseline-level already). Will update Table I when converged.
2. **Stronger case-study framing**: Added explicit disclaimer paragraph in Introduction
3. **Prior work comparison table** (Tab. II): Added to experiments section

## Ongoing Training
- `log/s32_v12_camgps_kd/`: CamGPS-only-KD training in progress (PID 271289, GPU 1)
- Early results: Ep0 Test DBA=0.8079, Ep3 Test DBA=0.8150 — tracking baseline DBA
- When complete: add row to Table I and update paper

## PDFs
- `main_round0_original.pdf` — Paper at start of this session (after figure fixes)
- `main_round1.pdf` — After Round 3 fixes (CamGPS CI, beam overhead, coherence time, DBA-gain)
- `main_round2.pdf` — After Round 4 fixes (case-study framing, prior work table)
- `main.pdf` — Current (= main_round2.pdf)
