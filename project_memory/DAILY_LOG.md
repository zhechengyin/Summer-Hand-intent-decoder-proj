# Daily Engineering Log

Last updated: 2026-07-04

Purpose: append what was done each day: commands, code changes, artifacts,
results, interpretation, and decisions. Do not use this as the short project
summary; keep that in [SUMMARY.md](SUMMARY.md).

## 2026-07-04

### LOG-001 - Fused Classical Baselines

Question: what is the honest fused baseline for the 4-class task?

Method: bandpower + fNIRS + LDA, FBCSP + fNIRS + LDA, and Riemannian tangent
space + fNIRS + Logistic Regression.

Artifacts:

- `results/metrics/metrics.json`
- `results/metrics/metrics_fbcsp.json`
- `results/metrics/metrics_riemannian.json`

Results:

- Bandpower + fNIRS + LDA: subject-CV 0.2404, LORO 0.2393.
- FBCSP + fNIRS + LDA: subject-CV 0.2201, LORO 0.2357.
- Riemannian + fNIRS + LogReg: subject-CV 0.2476, LORO 0.2715.

Interpretation: Riemannian is the best classical direction so far, but the gain
is modest. FBCSP likely fails because the four imagined actions are same-limb
movements with similar spatial covariance and frequency patterns.

Decision: keep Riemannian as the advanced baseline to tune first.

### LOG-002 - Neural Architecture Checks

Question: does adding model capacity improve the fused decoder?

Method: small GELU MLP, raw temporal CNN, 10-sample and 50-sample smoothed
temporal CNN, bare SNN, static Riemannian-SNN, and windowed Riemannian-SNN.

Artifacts:

- `results/metrics/metrics_temporal_cnn.json`
- `results/metrics/metrics_smoothed_temporal_cnn.json`
- `results/metrics/metrics_snn.json`
- `results/metrics/metrics_riemannian_snn.json`
- `results/metrics/metrics_windowed_riemannian_snn.json`

Results:

- Raw temporal CNN: subject-CV 0.2548, LORO 0.2596.
- 10-sample moving-average CNN: subject-CV 0.2559, LORO 0.2608.
- 50-sample moving-average CNN: subject-CV 0.2631, LORO 0.2667.
- Bare SNN: subject-CV 0.2535, LORO 0.2630.
- Static Riemannian-SNN: subject-CV 0.2512, LORO 0.2440.
- Windowed Riemannian-SNN: subject-CV 0.2428, LORO 0.2465.

Interpretation: more neural capacity did not clearly solve the task. The
50-sample smoother gave the best CNN result, but it still did not clearly beat
the Riemannian LORO baseline.

Decision: do not jump to larger neural nets without stronger evidence.

### LOG-003 - Symbolic ERD/ERS Feature Test

Question: can hand-designed ERD/ERS timing states expose class patterns better
than raw temporal learning?

Method: wide EEG epochs from -2 to 5 s, per-trial pre-onset baseline
normalization, region-level mu/low-beta/high-beta ERD/ERS state and timing
features, fused fNIRS hemodynamic features, LDA classifier.

Artifacts:

- `src/symbolic_erd.py`
- `results/metrics/metrics_symbolic_erd.json`
- `results/figures/confusion_fused_symbolic_erd_subject.png`
- `results/figures/confusion_fused_symbolic_erd_loro.png`

Results: subject-CV 0.2548, LORO 0.2655.

Interpretation: symbolic ERD/ERS is above chance and more class-balanced than
the CNN, but it remains near chance and below the best Riemannian LORO result.

Decision: keep as a promising feature family, but report as modest evidence only.

### LOG-004 - Riemannian Geometry Visualization And K-means

Question: do tangent-space features visibly separate Reach, Grasp, Lift, Twist?

Method: tangent-space projections, K-means probes, per-subject plots, and
covariance-aligned projection plots.

Artifacts:

- `tools/riemannian_cluster_probe.py`
- `tools/aligned_riemannian_plot.py`
- `results/metrics/riemannian_cluster_probe.json`
- `results/metrics/aligned_riemannian_tangent_plot.json`
- `results/metrics/aligned_riemannian_tangent_plot_ma50.json`

Results:

- EEG tangent space K-means: best k 7, class NMI 0.0001, subject NMI 0.9929.
- EEG tangent space + fNIRS K-means: best k 2, class NMI 0.0024.

Interpretation: the Riemannian manifold is dominated by subject/domain
structure, not the four action classes. Visual projections do not show reliable
Reach/Grasp/Lift/Twist separation.

Decision: tune Riemannian carefully, but assume the bottleneck is class
separability and signal quality unless new evidence appears.

### LOG-005 - Subject-ID Diagnostic

Question: are Riemannian EEG features mainly learning who the subject is?

Method: train a small NN to classify subject identity from Riemannian tangent
features.

Artifacts:

- `tools/riemannian_subject_id_probe.py`
- `results/metrics/subject_id_riemannian_nn_probe.json`

Results:

- EEG tangent space -> NN: subject-CV 1.0000, LORO 0.9990.
- EEG tangent space + fNIRS -> NN: subject-CV 0.9950, LORO 0.9870.

Interpretation: Riemannian features are extremely informative for subject
identity. This supports the idea that covariance structure carries strong
personal signatures but weak action-class signatures in this dataset.

Decision: consider alignment/normalization methods, but verify against action
accuracy rather than visual appeal.

### LOG-006 - fNIRS Waveform Inspection

Question: what does the fNIRS signal look like by class?

Method: plot HbO/HbR trial waveforms and class averages for the fNIRS epochs.

Artifacts:

- `results/figures/fnirs_hbo_hbr_class_average.png`
- `results/figures/fnirs_waveform_reach_examples.png`
- `results/figures/fnirs_waveform_grasp_examples.png`
- `results/figures/fnirs_waveform_lift_examples.png`
- `results/figures/fnirs_waveform_twist_examples.png`

Interpretation: fNIRS measures slow hemodynamic oxygenation changes, not direct
electrical spikes. It may add slow contextual signal, but it can also be weakly
aligned with fast same-limb motor imagery differences.

Decision: keep fNIRS in official fused N1 models, but use regularization or
feature selection so classifiers can ignore unhelpful fNIRS dimensions.

### LOG-007 - EEG-only Diagnostic Ablation

Question: is fNIRS damaging recent models?

Method: rerun selected recent feature families on the same fNIRS-aligned trials,
but drop fNIRS features. This is a diagnostic ablation only.

Artifacts:

- `tools/eeg_only_ablation.py`
- `results/metrics/eeg_only_ablation.json`

Results:

- EEG-only bandpower + LDA: subject-CV 0.2429, LORO 0.2201.
- EEG-only Riemannian + LogReg: subject-CV 0.2668, LORO 0.2715.
- EEG-only symbolic ERD/ERS + LDA: subject-CV 0.2488, LORO 0.2655.

Interpretation: fNIRS does not globally damage accuracy. It appears harmful for
Riemannian subject-CV, helpful for bandpower LORO, mildly helpful for symbolic
ERD/ERS subject-CV, and mostly irrelevant for the best Riemannian LORO score.

Decision: do not remove fNIRS globally. Instead, test better fusion,
regularization, and feature selection.

### LOG-008 - Memory Split

Question: should project memory be a clean summary or a daily engineering log?

Method: split the previous single `PROJECT_MEMORY.md` into a root index,
`project_memory/SUMMARY.md`, and `project_memory/DAILY_LOG.md`.

Artifacts:

- `PROJECT_MEMORY.md`
- `project_memory/SUMMARY.md`
- `project_memory/DAILY_LOG.md`

Decision: keep stable state in `SUMMARY.md`; append daily work to
`DAILY_LOG.md`.

### LOG-009 - Next-Round Experiments (PLS, shrinkage, connectivity, adversarial)

Question: can supervised latent geometry (PLS), covariance shrinkage,
functional connectivity, or subject-adversarial suppression beat the Riemannian
+ fNIRS baseline (LORO 0.2715)?

Method: new self-contained harness `tools/next_experiments.py` reusing the
cached, aligned EEG+fNIRS epochs (836 trials, 7 subjects) and the same
subject-specific K-fold + leave-one-run-out (LORO) protocols. All trainable
state fit on train split only. Chance = 0.25.

Command:
- `py tools/next_experiments.py --exp baseline shrinkage pls`
- `py tools/next_experiments.py --exp connectivity multiview`
- `py tools/next_experiments.py --exp adversarial`

Artifacts:
- `tools/next_experiments.py`
- `results/metrics/next_experiments.json` (baseline)
- `results/metrics/next_shrink_pls.json`
- `results/metrics/next_conn.json`
- `results/metrics/next_adv.json`

Results (LORO = strict/headline metric; subj = subject-specific K-fold):

| Model | subj | LORO |
| --- | ---: | ---: |
| Riemannian tangent + fNIRS + logreg (baseline, reproduced) | 0.2476 | 0.2715 |
| Tangent (Ledoit-Wolf cov) + fNIRS + logreg | 0.2572 | 0.2763 |
| Tangent (OAS cov) + fNIRS + logreg | 0.2464 | 0.2715 |
| Tangent + fNIRS -> PLS-DA(2) -> LDA | 0.2346 | 0.2799 |
| Connectivity wPLI (all ch) + fNIRS + logreg | 0.2715 | 0.2762 |
| Connectivity PLV+imcoh+wPLI (all ch) + fNIRS + logreg | 0.2596 | **0.2965** |
| Connectivity PLV+imcoh+wPLI (motor ch) + fNIRS + logreg | 0.2608 | 0.2715 |
| Multiview: tangent + connectivity + fNIRS + logreg | 0.2656 | 0.2906 |
| LOSO cross-subject plain MLP (diagnostic) | - | 0.2691 |
| LOSO cross-subject subject-adversarial MLP (diagnostic) | - | 0.2608 |

Shrinkage `reg` sweep detail: reg 1e-4..1e-1 trades subject-CV for LORO. Higher
reg helps subject-CV (reg=0.1 -> subj 0.264) but hurts LORO; best LORO at the
default 1e-3. Ledoit-Wolf auto-shrinkage is the best fixed-family choice.

Interpretation:
- Functional connectivity is the strongest new direction. All-channel
  PLV+imcoh+wPLI + fNIRS reaches LORO 0.2965 (bal 0.297, macro-F1 0.296,
  class-balanced), the best run-held-out score to date, ~+0.025 over the prior
  best and roughly doubling the above-chance margin (0.046 vs 0.021). This
  matches the same-limb-MI literature: the four actions differ more in network
  phase coordination than in channel bandpower/covariance. wPLI-only (0.2762)
  and the multiview combo (0.2906) corroborate the direction.
- PLS-DA with only 2 latent components gives the best *pure-tangent* LORO
  (0.2799) while dropping subject-CV to 0.235: it compresses to a couple of
  run-stable, label-aligned directions, consistent with "too many covariance
  dimensions, too little class signal."
- Subject-adversarial suppression FAILED to help (0.2608 vs 0.2691 plain). And
  the plain cross-subject LOSO (0.2691) is already ~= within-subject LORO
  (0.2715). So subject identity being decodable does NOT mean removing it
  exposes class signal; the bottleneck is class separability itself, not
  subject nuisance variance. Honest negative result for the adversarial idea.

Caveats: all-channel connectivity is 918-dim on ~2-run LORO training folds;
only LORO improved, not subject-CV. Treat 0.2965 as a promising, still-near-
chance lead that needs regularization tuning / replication, not a solved task.

Not run (deferred, need external assets or high overfit risk on 836 trials):
foundation embeddings (MIRepNet/NeurIPT), MIND-dataset pretraining, and a
from-scratch spatial-spectral transformer / BiCAT.

Decision: promote functional connectivity to a first-class EEG feature family;
next, regularize it (feature selection / lower-C logreg / shrinkage LDA) and
sweep bands, then try connectivity + PLS. Drop the subject-adversarial line.

### LOG-010 - Combining The Winning Ingredients

Question: can the LOG-009 winners be combined to increase results?

Method: three mechanistically-motivated combinations in
`tools/next_experiments.py --exp combos` (connectivity recomputed under a new
memory-safe chunked/complex64 implementation): (1) connectivity -> PLS-DA
compression -> LDA; (2) probability-level (late) fusion of connectivity +
Riemannian tangent; (3) connectivity -> shrinkage-LDA. Same subject-specific +
LORO protocols. Chance = 0.25.

Artifacts: `results/metrics/next_combos.json`.

Results:

| Model | subj | LORO |
| --- | ---: | ---: |
| Connectivity (all) + fNIRS + logreg [replicated] | 0.2596 | 0.2965 |
| Connectivity -> PLS-DA(8) -> LDA + fNIRS | 0.2715 | 0.2954 |
| Connectivity -> PLS-DA(16) -> LDA + fNIRS | 0.2727 | 0.2858 |
| Connectivity -> PLS-DA(32) -> LDA + fNIRS | 0.2595 | 0.2751 |
| Connectivity -> shrinkage-LDA + fNIRS | 0.2668 | 0.2870 |
| Late fusion connectivity(0.5)+tangent + fNIRS | 0.2584 | 0.2942 |
| Late fusion connectivity(0.8)+tangent + fNIRS | 0.2632 | 0.2894 |

Interpretation:
- The chunked complex64 connectivity reproduced LORO 0.2965 exactly, confirming
  the LOG-009 headline is stable, not a numerical artefact.
- No combination raised the LORO ceiling above raw connectivity (0.2965). BUT
  connectivity -> PLS-DA(8) is a strictly better *model*: it matches the LORO
  (0.2954) while lifting subject-CV from 0.260 to 0.272 and compressing 918
  features to 8 supervised latent dims. That removes the main caveat
  (high-dimensional, subj-CV flat) and makes the result far more trustworthy.
- Connectivity -> shrinkage-LDA (0.267 / 0.287) is a good simpler regularized
  alternative that also moves both protocols together.
- Late fusion with the tangent model does NOT beat connectivity alone (best
  0.2942 at w=0.5). Combined with the earlier diluted multiview, this confirms
  the Riemannian tangent adds little on top of connectivity; do not spend more
  effort fusing the two.

Decision: adopt connectivity -> PLS-DA(8..16) -> LDA as the connectivity model
of record (robust, low-dim). Regularize/ band-sweep from there. Stop trying to
combine connectivity with the tangent front end.

### LOG-011 - Cross-Dataset Validation on a Decodable MI Contrast (ds004362)

Question: do our best ds004022 front ends extract real signal when a genuinely
decodable MI contrast exists? (Sanity check that ds004022's near-chance is task
difficulty, not a pipeline bug.)

Method: new second OpenNeuro dataset ds004362 = PhysioNet EEG Motor
Movement/Imagery (eegmmidb), 64-ch EEGLAB .set @ 160 Hz. Per its README, runs
4/8/12 = imagined fist movement, event value ...T1 = LEFT fist, ...T2 = RIGHT
fist. Decoded LEFT vs RIGHT hand motor imagery (classic lateralized MI,
chance = 0.50). Downloaded runs 4/8/12 for 5 subjects (sub-001..sub-005) over
public HTTPS (S3), epoched 0.5-3.5 s after cue on 21 sensorimotor channels
(FC/C/CP rows), and applied the SAME front ends and CV protocols as ds004022
(EEG-only; no fNIRS in this dataset).

Command: `py tools/eegmmidb_probe.py`

Artifacts:
- `tools/eegmmidb_probe.py`
- `results/metrics/eegmmidb_probe.json`
- `data/ds004362/` (5 subjects x runs 4/8/12, git-ignored)

Data: 225 trials, 21 motor ch x 480 samp, ~113 left / 112 right, 5 subjects.

Results (chance = 0.50):

| Model | subject-CV | LORO |
| --- | ---: | ---: |
| Riemannian tangent + logreg | 0.658 | **0.707** |
| Connectivity PLV+imcoh+wPLI + logreg | 0.556 | 0.573 |
| Connectivity -> PLS(8) -> LDA | 0.547 | 0.564 |

Interpretation:
- PIPELINE VALIDATED. On a genuinely decodable MI dataset the exact same
  Riemannian tangent front end reaches 0.71 LORO / 0.66 subject-CV, well above
  chance and in the normal range for left/right MI on ~21 channels with few
  trials. So ds004022's near-chance 4-class same-limb result is task difficulty,
  not an implementation bug.
- The BEST model is task-dependent, and it makes physiological sense:
  * Left/right MI is a spatial lateralization contrast (C3 vs C4 mu/beta power),
    which channel covariance / tangent space captures directly -> tangent wins.
  * Connectivity (phase coupling / network coordination) is only weakly above
    chance here because the dominant signal is lateralized power, not coupling.
  * On ds004022 same-limb MI there is no lateralization to exploit, so tangent
    had little to grab and connectivity's network view was relatively best.
  This is a coherent story: covariance-geometry for lateralized MI, functional
  connectivity for same-limb MI.

Caveats: only 5 subjects, ~15 trials/class/subject, 21 channels; numbers are a
probe, not a tuned benchmark. eegmmidb left/right MI typically lands ~0.65-0.80
subject-specific, consistent with what we see.

Decision: keep ds004362 as a standing positive-control dataset. When changing
the EEG front end, sanity-check it here (should stay well above 0.50) before
trusting near-chance moves on ds004022.

### LOG-012 - Architecture Benchmark: Classical vs CNN vs Transformer

Question: how do our best classical/geometric front ends stack up against top
deep MI architectures (EEGNet-style CNN, EEG Conformer transformer)?

Method: new compact EEG Conformer (`src/conformer.py`) with the SAME training
recipe as the temporal CNN (per-channel standardize, decimate, AdamW, early
stop) so the comparison isolates architecture. Benchmark driver
`tools/architecture_bench.py` runs every model under the identical
subject-specific + LORO protocol on both datasets. Deep nets kept small (small
data). Also patched `src/temporal_cnn.py` to accept 0 fNIRS features (EEG-only).

Command:
- `py tools/architecture_bench.py --dataset eegmmidb`
- `py tools/architecture_bench.py --dataset ds004022`

Artifacts:
- `src/conformer.py`, `tools/architecture_bench.py`
- `results/metrics/architecture_bench_eegmmidb.json`
- `results/metrics/architecture_bench_ds004022.json`

Results, ds004362 left/right hand MI (EEG-only, chance 0.50):

| Model | subject-CV | LORO |
| --- | ---: | ---: |
| Riemannian tangent + logreg | 0.658 | **0.707** |
| Connectivity PLV+imcoh+wPLI + logreg | 0.556 | 0.573 |
| EEGNet-style CNN | 0.516 | 0.551 |
| EEG Conformer (transformer) | 0.471 | 0.507 |

Results, ds004022 4-class same-limb (fused EEG+fNIRS, chance 0.25):

| Model | subject-CV | LORO |
| --- | ---: | ---: |
| Riemannian tangent + logreg | 0.248 | 0.272 |
| Connectivity -> PLS(8) -> LDA | 0.272 | **0.295** |
| EEGNet-style CNN (fused) | 0.261 | 0.258 |
| EEG Conformer (fused) | 0.243 | 0.232 |

Literature context (reported, NOT our data): on BCI IV-2a (4-class, 288
trials/subject) EEGNet ~70%, EEG Conformer ~78.7%, ATCNet ~85%, CTNet ~82.5%,
recent two-stage transformer ~88.5%. Those numbers rely on ~10x more trials per
subject plus augmentation.

Interpretation:
- On BOTH datasets the CNN and the transformer FAIL TO BEAT our classical /
  geometric best, and the transformer is the weakest model overall (near or
  below chance). This is the expected small-data regime: Conformer-class models
  need hundreds of trials/subject + augmentation (their SOTA numbers) to shine;
  here they see ~30-115 trials per training fold and cannot learn, while
  Riemannian tangent + logreg is a strong low-data estimator.
- ds004362 confirms it cleanly: on a genuinely decodable contrast the ranking is
  Riemannian (0.71) > connectivity (0.57) > CNN (0.55) > Transformer (0.51).
- ds004022 confirms our earlier conclusion: even a transformer cannot
  manufacture 4-class same-limb signal that is not linearly present; connectivity
  -> PLS remains the best at 0.295.

Decision: do not pursue larger transformers for these datasets. Our compact
classical/geometric pipeline is the right tool at this data scale; a transformer
would only be justified with a much larger dataset or transfer/pretraining
(e.g., the deferred MIRepNet / MIND directions).

### LOG-013 - Connectivity Band/Metric Sweep + 10-Subject Benchmark

Question: (a) can a band/metric sweep improve the connectivity -> PLS model of
record? (b) does the architecture ranking hold with more subjects?

Method:
- `py tools/next_experiments.py --exp connsweep` : sweep frequency band sets and
  metric subsets feeding conn -> PLS(k) -> LDA on ds004022 (fused).
- `py tools/architecture_bench.py --dataset eegmmidb --subjects 10` : re-run the
  classical-vs-deep benchmark with 10 subjects (450 trials) instead of 5. Added
  robustness (skip subjects with bad runs) and a --subjects flag.

Artifacts:
- `results/metrics/next_connsweep.json`
- `results/metrics/architecture_bench_eegmmidb.json` (now 10 subjects)

Connectivity band/metric sweep, ds004022 (conn -> PLS(8) -> LDA, chance 0.25):

| Band set / metrics | subj | LORO |
| --- | ---: | ---: |
| mu+beta [8-13,13-30] all 3 metrics | 0.272 | **0.295** |
| fine [8-13,13-20,20-30] all 3 metrics | 0.251 | 0.279 |
| beta [13-30] all 3 | 0.264 | 0.268 |
| high-beta [20-30] all 3 | 0.264 | 0.261 |
| low-beta [13-20] all 3 | 0.256 | 0.261 |
| mu [8-13] all 3 | 0.222 | 0.257 |
| fine [plv+wpli] | 0.254 | 0.290 |
| fine [wpli] | 0.261 | 0.286 |
| fine [plv] / [imcoh] | 0.245 / 0.230 | 0.264 / 0.262 |
| fine all -> PLS(4/12/16) | ~0.23-0.25 | 0.278 / 0.270 / 0.278 |

-> The existing config (mu+beta, all 3 metrics, PLS(8)) is confirmed OPTIMAL;
nothing in the sweep beat LORO 0.2954. Two-band mu+beta beats the 3-band fine
split (finer split adds dims PLS can't compress as cleanly). No change to the
model of record.

Architecture benchmark, ds004362 left/right MI, 10 subjects / 450 trials
(EEG-only, chance 0.50):

| Model | subject-CV | LORO |
| --- | ---: | ---: |
| Riemannian tangent + logreg | 0.629 | **0.642** |
| EEGNet-style CNN | 0.533 | 0.569 |
| Connectivity PLV+imcoh+wPLI | 0.576 | 0.558 |
| EEG Conformer (transformer) | 0.516 | 0.556 |

Interpretation: with 2x the subjects the ranking holds and is more trustworthy.
Riemannian tangent remains clearly best (0.64 LORO). The deep nets improved
slightly with more data (CNN LORO 0.569 now edges connectivity) and the
transformer's LORO caught up to the CNN/connectivity cluster, but neither
approaches Riemannian, and the transformer is still weakest on subject-CV
(0.516). Confirms: at this data scale the classical/geometric pipeline wins;
deep nets would need far more data (their BCI IV-2a SOTA uses ~10x trials).
The earlier 5-subject Riemannian number (0.707) was optimistic; 0.64 on 10
subjects is the more honest estimate.

Decision: keep mu+beta/all-metrics/PLS(8) as the connectivity model of record;
keep Riemannian tangent as the go-to for lateralized MI. Deep nets parked until
a transfer/pretraining path (MIRepNet/MIND) or a larger dataset is available.

### LOG-014 - Cross-Domain Validation on Consumer EEG (Muse2, safetyai repo)

Question: does our best pipeline generalise to a totally different device/task
(4-channel Muse2 consumer EEG, eye/blink/idle states) in the local safetyai
StreamAdapt repo?

Method: new probes `tools/muse_probe.py` (root single-session recordings) and
`tools/cursor_probe.py` (CursorSelectionData, 4 subjects x 4 classes
background/blink/left/right x5 recordings -> proper leave-one-RECORDING-out).
EOG is low-frequency so the tangent front end uses a broadband 1-40 Hz filter
(not the 8-30 MI band). Chance 0.25.

Artifacts: `tools/muse_probe.py`, `tools/cursor_probe.py`,
`results/metrics/muse_probe_Alex.json`, `results/metrics/cursor_probe_all.json`.

Results:
- Root Alex 4-class (single session/class): tangent+logreg chrono 0.847;
  tangent + connectivity[plv+imcoh+wpli] multiview chrono 0.889 (best). k-fold
  optimistic ~0.90-0.93 but flagged as leaky (one recording/class).
- CursorSelectionData ALL 4 subjects, leave-one-recording-out (leakage-free):
  Riemannian tangent 0.767 LORO (0.807 subj) > tangent+connectivity 0.734 >
  connectivity alone 0.550. Chance 0.25.

Interpretation: pipeline transfers cleanly to 4-ch consumer EEG (~0.77
leakage-free). Tangent geometry captures the EOG amplitude/spatial structure;
connectivity is weak alone (phase coupling is not the EOG signal) and, under the
rigorous protocol, dilutes tangent (mirrors the ds004022 early-concat finding).
The single-session multiview gain was a small-sample effect.

### LOG-015 - Two-Stage Identify-Then-Decode Router

Question: exploit "Riemannian identifies the subject almost perfectly" -- use a
Stage-1 subject-ID router, then Stage-2 that subject's own decoder. Does it beat
a pooled model when identity is unknown at test time?

Method: `tools/two_stage.py`. Cohort-level leave-one-run-out. Stage 1 = tangent
-> logreg -> subject id; Stage 2 = predicted subject's LDA class decoder. Compare
pooled (ignore id) / two-stage (predicted route) / oracle (true route = ceiling).

Artifacts: `tools/two_stage.py`, `results/metrics/two_stage_{cursor,eegmmidb,ds004022}.json`.

Results:

| Dataset (chance) | Stage-1 subj-ID | pooled | two-stage | oracle |
| --- | ---: | ---: | ---: | ---: |
| cursor 4-class, 4 subj (0.25) | 0.786 | 0.637 | 0.695 | 0.769 |
| eegmmidb L/R, 10 subj (0.50) | 1.000 | 0.569 | 0.609 | 0.609 |
| ds004022 same-limb, 7 subj (0.25) | 1.000 | 0.266 | 0.260 | 0.260 |

Interpretation: Stage-1 subject-ID is excellent (1.00 on both MI datasets, 0.79
on the harder leave-one-recording-out Muse case). The router HELPS exactly when
per-subject decoders beat a pooled one, bounded by (oracle - pooled) and scaled
by ID accuracy: cursor +5.8, eegmmidb +4.0 (==oracle since ID perfect). On
ds004022 routing is perfect but there is no per-subject class advantage to
capture (class signal absent) -- pooled is marginally better. A genuinely useful
technique for unknown-identity deployment on decodable tasks; validates turning
the subject-separation finding into accuracy.

### LOG-016 - WAY-EEG-GAL Grasp-and-Lift: Top-3 Techniques + Activation/Optimizer Study

Question: how do our top-3 techniques do on a 3rd, richer dataset (WAY-EEG-GAL,
32-ch EEG @ 500 Hz, grasp-and-lift), and which activation/optimizer is best for
the neural technique?

Method: downloaded participants P1-P3 from figshare collection 988376 (each an
~800 MB zip: HS/WS mat files + AllLifts). Parser `tools/way_gal_probe.py` reads
per-trial windowed EEG from WS_P*_S*.mat (ws.win(n).eeg, .eeg_t, .LEDon,
.weight, .surf). Two tasks: move-vs-rest (peri-LED vs pre-LED baseline, chance
0.50, positive control) and weight (165/330/660 g, chance 0.33, hard). Weight
restricted to weight-VARYING series (surface-varying series hold weight constant
-> would confound). Subject-specific K-fold + leave-one-SERIES-out. Techniques:
(1) Riemannian tangent + LDA, (2) connectivity[plv+imcoh+wpli] -> PLS(8) -> LDA,
(3) MLP on tangent features (AdamW) sweeping activation.

Artifacts: `tools/way_gal_probe.py`, `results/metrics/way_gal_{move_rest,weight}.json`.

Results (P1+P2+P3, per-subject averaged):

move-vs-rest (chance 0.50, 1764 trials):
| Technique | subj | LORO |
| --- | ---: | ---: |
| MLP tangent (relu/leaky_relu, AdamW) | 0.832 | 0.830 |
| MLP tangent (gelu / silu / elu / tanh) | 0.820-0.829 | 0.820-0.829 |
| Connectivity -> PLS -> LDA | 0.731 | 0.722 |
| Riemannian tangent + LDA | 0.652 | 0.561 |

weight (chance 0.33, 678 trials, weight-varying series):
| Technique | subj | LORO (bal) |
| --- | ---: | ---: |
| Connectivity -> PLS -> LDA | 0.360 | 0.370 (0.358) |
| MLP tangent (silu) | 0.355 | 0.351 (0.327) |
| Riemannian tangent + LDA | 0.330 | 0.301 (0.292) |

Interpretation:
- Positive control is strongly decodable: MLP on tangent ~0.83 LORO, connectivity
  0.72. Pipeline extracts real movement signal on a 3rd device/dataset.
- Weight (fine intent parameter) is ~chance (0.33-0.37) across all techniques --
  same lesson as same-limb MI: gross movement easy, subtle intent parameters
  hard. Honest and consistent.
- The MLP hugely out-generalises tangent+LDA on the strong task (LORO 0.830 vs
  0.561): with abundant high-SNR signal the neural head wins -- opposite of the
  tiny MI datasets where deep nets lost. Deep nets pay off only with enough signal.
- Activation study (AdamW throughout): all within ~1% (relu/leaky_relu 0.832 ~
  gelu 0.829 ~ silu 0.825 ~ elu/tanh 0.820). ReLU/LeakyReLU edge GELU by noise;
  GELU is a fine default. Optimizer AdamW (decoupled weight decay) with lr 1e-3,
  wd 1e-3, hidden [128,64], dropout 0.4, early stop -- solid defaults; activation
  is not where the accuracy lives. Weight task is signal-limited, not
  hyperparameter-limited.

### LOG-017 - TCN+GRU Sequence Model on WAY-EEG-GAL

Question: does a Temporal Convolutional Network + GRU (dilated causal conv ->
GRU -> head) on raw EEG beat the feature-based techniques?

Method: `src/tcn_gru.py` (spatial pointwise mix -> dilated causal TCN residual
blocks, dilations 1/2/4/8 -> GRU -> linear head), same trainer recipe (decimate
/5, AdamW, early stop). `tools/tcn_gru_bench.py`. Run on P1, both tasks.

Artifacts: `src/tcn_gru.py`, `tools/tcn_gru_bench.py`,
`results/metrics/way_gal_tcngru_P1.json`.

Results (P1):
- move-vs-rest: subj 0.944 / LORO 0.908  (~900 s/run)
- weight:       subj 0.381 / LORO 0.319 (bal 0.285)  -- near chance
- params: 65,378 (vs MLP-on-tangent 76,098)

Interpretation: TCN+GRU is competitive (marginally higher subj-CV 0.944 vs the
MLP's 0.940) but SLIGHTLY WORSE on the honest LORO (0.908 vs 0.935), at ~30x the
compute (sequential conv+recurrence over 150 time steps vs an MLP on a 528-dim
tangent vector). Fewer params yet far slower -- cost is sequential compute, not
parameter count. Near chance on weight like everything else. The sophisticated
sequence model does NOT beat the tiny tangent+MLP: feature-based geometry + a
small head remains the right tool at this data scale.

## Entry Template

### LOG-XXX - Short Name

Question:

Method:

Command:

Artifacts:

Results:

Interpretation:

Decision:
