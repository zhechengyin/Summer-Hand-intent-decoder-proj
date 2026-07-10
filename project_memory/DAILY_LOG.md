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

### LOG-018 - WAY-EEG-GAL EEG->Kinematics: Decode Hand/Finger Velocity

Question: can we decode continuous hand/finger VELOCITY from EEG (the canonical
"movement through brain signals" use of the kinematics block)?

Method: `tools/way_gal_kinematics.py`. The kin block has 3D positions of 4
markers (Px/Py/Pz sensors 1-4) on the same 500 Hz grid as EEG. Pipeline: EEG
low-pass 4 Hz (movement-related slow potentials) -> decimate to 25 Hz -> +/-240 ms
time-lagged design matrix -> Ridge regression -> per-axis velocity. Target =
gradient of low-passed position. Scored by Pearson r (pred vs true velocity)
under leave-one-series-out. (First attempt OOM'd on the RAM-tight box: fixed with
decim 20 / 6 lags / float32 in-place standardisation.)

Artifacts: `tools/way_gal_kinematics.py`, `results/metrics/way_gal_kin_P1.json`.

Results (P1, leave-one-series-out):
| Marker | r_mean | r_x | r_y | r_z |
| --- | ---: | ---: | ---: | ---: |
| sensor 1 (reference/object, ~static) | 0.289 | 0.286 | 0.171 | 0.409 |
| sensor 2 (hand/finger) | 0.530 | 0.586 | 0.650 | 0.353 |
| sensor 3 (hand/finger) | 0.569 | 0.670 | 0.677 | 0.360 |
| sensor 4 (hand/finger) | 0.583 | 0.681 | 0.670 | 0.398 |

Interpretation: continuous EEG->velocity decoding works well -- r ~ 0.53-0.58
mean, up to 0.68 per axis for the moving hand/finger markers, at or above typical
published EEG kinematics numbers (r ~ 0.3-0.5). A genuinely different capability
from the classification tasks: this dataset supports regression of movement
trajectory, not just event/parameter classification.

Caveat (honest): low-frequency all-channel EEG->velocity decoding can be partly
driven by movement/EOG/EMG artifacts correlated with the motion, not purely
motor-cortical signal (known debate in the literature). r~0.58 is a real standard
result but an artifact-controlled version (ICA cleaning, central-motor-channel
restriction) is needed before claiming it is purely cortical.

### LOG-019 - EEG->Velocity: Full-Capacity + Artifact-Controlled (P1-P3)

Question: was r~0.58 the ceiling, and how much survives artifact control?

Method: `tools/way_gal_kin_full.py`. Memory-safe high capacity via per-SERIES
X^T X / X^T Y accumulation (never materialise the full design matrix -> no OOM);
finer resolution (50 Hz, +/-240 ms lags); ridge alpha tuned per outer fold on an
inner validation series. Two channel sets: ALL (ceiling) vs MOTOR-only (drop
frontal EOG + temporal EMG -> artifact control). Leave-one-series-out, Pearson r.

Artifacts: `tools/way_gal_kin_full.py`,
`results/metrics/way_gal_kin_full_P1_P2_P3.json`.

Results (best hand/finger marker, r_mean):
| Subject | all-channel | motor-only |
| --- | ---: | ---: |
| P1 | 0.622 (axis up to 0.72) | 0.427 (up to 0.48) |
| P2 | 0.351 | 0.271 |
| P3 | 0.407 | 0.308 |
| mean | ~0.46 | ~0.33 |

Interpretation:
- Tuned full-capacity BEAT the reduced first pass (P1 0.622 vs 0.583) -> r~0.58
  was NOT the ceiling.
- Strong subject variability: all-channel r 0.35-0.62 (P1 an unusually good
  subject); ~0.46 mean.
- Artifact control matters: motor-only drops ~0.10-0.20 (mean 0.46 -> 0.33), so a
  real fraction of the all-channel number rides on non-motor (EOG/EMG-prone)
  channels. BUT motor-only still decodes r~0.27-0.43 (mean 0.33) -- genuine
  central-motor velocity signal survives, well above zero.
- Honest headline: EEG->hand/finger velocity is real (motor-only r~0.33, best
  subject 0.43), the naive all-channel number (~0.46, best 0.62) is partly
  artifact-inflated, and further gains are still possible (more lags, per-subject
  tuning, nonlinear/PLS regressors, ICA cleaning).

### LOG-020 - EEG->Velocity Decoder Benchmark: Linear vs Riemannian vs TCN+GRU

Question: can a Riemannian tangent method or a TCN+GRU beat the lagged-linear
Ridge for continuous velocity decoding?

Method: `tools/way_gal_kin_models.py`. Target = 3D velocity of best hand/finger
marker (sensor 4), EEG low-pass 4 Hz -> 50 Hz. Three decoders, all channels:
(1) lagged-linear Ridge (+/-240 ms lags) [leave-one-series-out];
(2) sliding-window Riemannian tangent (1 s windows, per-window covariance ->
tangent -> Ridge to window-centre velocity) [LOSO];
(3) seq2seq TCN+GRU (spatial conv -> dilated causal TCN dil 1/2/4/8 -> bi-GRU
-> per-timestep 3D head, GELU/AdamW, F=32) [3-fold over series]. Params guided by
light literature review (GRU-family strong for EEG kinematics; TCN task-dependent).

Artifacts: `tools/way_gal_kin_models.py`,
`results/metrics/way_gal_kin_models_P{1,2,3}.json`.

Results (marker 4, r_mean; per-subject):
| Subject | linear | tangent | TCN+GRU |
| --- | ---: | ---: | ---: |
| P1 | 0.603 | 0.471 | 0.788 |
| P2 | 0.339 | 0.339 | 0.625 |
| P3 | 0.384 | 0.267 | 0.580 |
| mean | 0.442 | 0.359 | 0.664 |

Interpretation:
- TCN+GRU WINS clearly on all 3 subjects (mean r 0.664, +0.22 over linear),
  best axes ~0.84 on P1. A recurrent seq2seq model captures temporal dynamics a
  static covariance or fixed-lag linear map cannot. Notably it wins despite the
  3-fold protocol giving it LESS training data (6 series) than the linear/tangent
  LOSO (8 series) -- so the gain is real, not a protocol artifact.
- Riemannian tangent (sliding window) UNDERperforms the linear baseline (0.359 vs
  0.442). Confirms tangent/covariance geometry is a per-trial CLASSIFICATION
  tool: for continuous regression it discards the instantaneous temporal detail
  velocity needs. So the Riemannian technique does not transfer to velocity.
- Caveat: all-channel; like the linear (which dropped ~0.62->0.43 motor-only in
  LOG-019), the TCN+GRU 0.664 is likely partly EOG/EMG-inflated. A motor-only
  TCN+GRU would be lower but the honest cortical number -- next step.

Decision: TCN+GRU is the velocity decoder of choice; drop the tangent approach
for regression. Run an artifact-controlled (motor-channel) TCN+GRU next.

### LOG-021 - Velocity Research Loop: Movement-Window Cropping + Pooling

Goal: push EEG->velocity correlation above the committed TCN+GRU baseline
(mean r 0.664 across P1-P3). Harness `tools/way_gal_kin_research.py`.

Speed fix: the old runs (50 Hz, uncropped 9.4 s, 80 ep) took ~1000 s/config
because the bidirectional GRU is sequential over long sequences. Switched to
25 Hz + crop to the movement window [1.5, 7.0] s + 30 ep + 4 threads ->
~120 s/config (~8x faster).

Results (marker 4, r_mean, 3-fold):
| Config | P1 | P2 | P3 | mean |
| --- | ---: | ---: | ---: | ---: |
| uncropped 50 Hz (committed baseline) | 0.788 | 0.625 | 0.580 | 0.664 |
| cropped 25 Hz within-subject (lp=2) | 0.830 | 0.655 | 0.757 | 0.747 |
| cropped 25 Hz POOLED (lp=2) | 0.850 | 0.590 | 0.811 | 0.750 |

Findings:
- CROPPING TO THE MOVEMENT WINDOW is the real lever: +0.083 mean (0.664 ->
  0.747). Removing the pre-cue rest period (where true velocity ~ 0 and only
  dilutes the correlation) sharply improves r AND speeds training. New best mean
  ~0.75; P1 axis up to 0.90.
- lp=2 Hz ~ lp=4 Hz (2 marginally better).
- CROSS-SUBJECT POOLING is a NET WASH vs within-subject cropped (0.750 vs
  0.747): it lifts weak subjects (P3 0.757->0.811) but hurts P2 (0.655->0.590).
  Not a general win; useful only for subjects with little signal.

Current best config: cropped 25 Hz, lp=2, TCN+GRU (F32, dil 1/2/4/8, bi-GRU H32),
within-subject; ~0.747 mean, pooling optional per subject.

Still to try: architecture (dilation16 context, GRU H64/L2), ensemble, and the
artifact-controlled motor-only version (honest cortical ceiling, expected lower).

### LOG-022 - Velocity Decoding BEST: BIG TCN+GRU, mean r 0.843

Question: does a larger temporal model on top of the cropping win push r higher?

Method: `tools/way_gal_kin_research.py --stage arch/final`. On the cropped 25 Hz
lp=2 config, sweep architecture then validate the best ("BIG") on P1-P3.
BIG = TCN dilations 1/2/4/8/16 (larger receptive field) + bidirectional GRU
hidden 64, 2 layers + F=64, 100 epochs, AdamW.

Arch sweep (P1): baseline 0.830 -> +dil16 0.856 -> +GRU H64/L2 0.858 ->
+F64 0.836 -> BIG(all) 0.889.

Final BIG, marker 4, r_mean (3-fold):
| Subject | r_mean | r_x | r_y | r_z |
| --- | ---: | ---: | ---: | ---: |
| P1 | 0.889 | 0.926 | 0.925 | 0.815 |
| P2 | 0.777 | 0.816 | 0.823 | 0.694 |
| P3 | 0.864 | 0.900 | 0.925 | 0.767 |
| MEAN | 0.843 | | | |

Progression of the whole research loop:
| Milestone | P1 | P2 | P3 | mean |
| --- | ---: | ---: | ---: | ---: |
| committed baseline (uncropped 50 Hz) | 0.788 | 0.625 | 0.580 | 0.664 |
| + movement-window crop (25 Hz, lp2) | 0.830 | 0.655 | 0.757 | 0.747 |
| + BIG arch | 0.889 | 0.777 | 0.864 | 0.843 |

Interpretation: EEG->hand/finger velocity decoding reaches mean r 0.843 (per-axis
up to 0.926) -- a strong result, well above typical published EEG kinematics
numbers (0.3-0.5). Two levers drove +0.18 over baseline: (1) cropping to the
movement window (removes rest-period dilution), (2) a bigger temporal model
(larger dilated-conv context + 2-layer bi-GRU). The hard subject P2 gained the
most from capacity (0.625 -> 0.777).

Caveat (unchanged): all-channel; part of r is likely EOG/EMG movement artifact
(the linear motor-only control was ~0.33 vs ~0.46 all-channel in LOG-019). A
motor-only BIG TCN+GRU would be the honest cortical ceiling -- still to run.

BEST config of record: cropped [1.5,7.0]s, 25 Hz, lp=2 Hz, BIG TCN+GRU.
Model size: 188,803 trainable params (~0.76 MB fp32): 64k conv/TCN front end +
124k bidirectional GRU (H64, 2 layers) + 387 head. (Baseline small model: 26k.)

### LOG-023 - Artifact-Controlled BIG TCN+GRU (motor channels only)

Question: how much of the BIG TCN+GRU velocity decoding (all-channel mean r
0.843) is genuine cortical signal vs EOG/EMG movement artifact?

Method: `tools/way_gal_kin_research.py --stage final_motor`. Same BIG config, but
restrict EEG to 17 central sensorimotor channels (F3/Fz/F4, FC*, C*, CP*, P3/Pz/
P4), dropping frontal EOG (Fp/F7/F8) and temporal EMG (T7/T8/TP9/TP10).

Results (marker 4, r_mean, 3-fold):
| Subject | all-channel | motor-only |
| --- | ---: | ---: |
| P1 | 0.889 | 0.851 |
| P2 | 0.777 | 0.763 |
| P3 | 0.864 | 0.855 |
| MEAN | 0.843 | 0.823 |

Interpretation (KEY): motor-only drops only 0.02 (0.843 -> 0.823), whereas the
LINEAR decoder dropped ~0.13 (0.46 -> 0.33 all-channel -> motor, LOG-019). So the
BIG TCN+GRU decodes largely GENUINE MOTOR-CORTICAL velocity signal, not mostly
artifact -- the nonlinear temporal model isolates a real cortical code the linear
map could not. This substantially strengthens the r=0.843 result: even the
conservative cortical estimate is r=0.823 (per-axis up to 0.91).

Note: P1's wall-timer read ~13 h because the laptop was suspended mid-run; the
computation is valid (P2/P3 ran in ~10-20 min).

Best result of record UPDATED: EEG->finger/hand velocity, BIG TCN+GRU, all-channel
mean r 0.843, motor-only (cortical) mean r 0.823.

### LOG-024 - Improved recipe (aug + cosine + longer context): mean r 0.853

Question: can cheap (param-light) levers push past BIG (mean r 0.843) while
staying < 1 MB (target inference device)?

Method: `--stage improve/final_improved`. BIGP = BIG + dilation 32 (longer
context) + data augmentation (additive Gaussian noise 0.1 + per-sample channel
dropout 0.1) + cosine LR schedule + 150 epochs. Param budget: 1 MB fp32 =
262,144 params.

Results (marker 4, r_mean, 3-fold):
| Subject | BIG (LOG-022) | BIGP |
| --- | ---: | ---: |
| P1 | 0.889 | 0.889 (saturated) |
| P2 | 0.777 | 0.813 (+0.036) |
| P3 | 0.864 | 0.858 (-0.006) |
| MEAN | 0.843 | 0.853 |

Model size: BIGP = 201,155 params (0.80 MB fp32) -- within 1 MB budget.

Interpretation: +0.010 mean, driven entirely by the hard subject P2 (+0.036) --
augmentation improves generalization where there is headroom; P1 is already at
ceiling (0.889) so it cannot gain, P3 unchanged. Modest but real, and free on the
memory budget. Longer context (dil32) alone did nothing on saturated P1; the win
is the augmentation.

NEW BEST OF RECORD: BIGP, all-channel mean r 0.853 (P1 0.889, P2 0.813, P3 0.858),
0.80 MB. Config: cropped [1.5,7]s, 25 Hz, lp=2, TCN dil 1/2/4/8/16/32, bi-GRU H64
L2, F64, noise 0.1 + chdrop 0.1, cosine, 150 ep.

### LOG-025 - Best Model Transfers to Intracortical NHP Reaching (finger velocity)

Question: does our best velocity decoder (TCN+GRU) transfer to a completely
different modality -- intracortical primate spikes instead of scalp EEG?

Dataset: O'Doherty et al. NHP reaching (Zenodo 3854034). Macaque M1/S1 Utah-array
spikes + fingertip position @ 250 Hz. Session indy_20161005_06 (smallest, 80 MB;
235 spiking units, 6.2 min). NOT EEG -- spiking units become the input channels.

Method: `tools/indy_velocity.py`. Bin spikes to 50 ms firing rates (20 Hz),
window into 2 s chunks (40 bins), decode 2D fingertip velocity (top-2 moving
axes) with the SAME TCN+GRU (dils 1/2/4/8/16, bi-GRU H64 L2, F64, aug+cosine),
leave-one-contiguous-block-out (5 blocks). Reuses build_net/run_nn.

Results (Pearson r):
| Model | r_mean | axis1 | axis2 | params |
| --- | ---: | ---: | ---: | ---: |
| TCN+GRU (our best) | 0.848 | 0.821 | 0.876 | 201,795 (0.81 MB) |
| lagged-linear ref | 0.731 | 0.677 | 0.785 | - |

Interpretation: the EEG-designed TCN+GRU transfers cleanly to intracortical
spikes -- finger-velocity r 0.848 with the same <1 MB model, beating the linear
decoder by +0.12. Confirms the architecture is modality-general (works on scalp
EEG voltage AND binned cortical spike rates as long as the input is
channels-x-time). Cross-modality demonstration.

Caveats: single session / one animal (indy), untuned for spikes (50 ms bins, 2 s
windows); intracortical SOTA is ~0.9+ with more data/tuning, so 0.848 is a strong
zero-tuning transfer, not a ceiling. Different modality from the EEG work -- keep
as a separate cross-modality result, not merged with the EEG velocity ledger.
Also fixed run_linear to infer output dim (was hardcoded to 3; failed on 2D).

### LOG-026 - NHP Cross-Session HELD-OUT Test (train sessions != test sessions)

Question: how well does the model generalise to data NOT in training? Correcting
LOG-025's overstated "can't pool across sessions": that is only true for SORTED
units (variable identity). PER-ELECTRODE multiunit counts give a consistent
96-channel space across indy's sessions, so we can pool sessions to train and
test on entirely HELD-OUT sessions.

Method: `tools/indy_crosssession.py`. Per-electrode rates (96 ch, sum all units
on each electrode), 50 ms bins, 2 s windows. Train on 6 indy sessions (pooled,
per-session z-scored), TEST on 2 sessions never in training. Best TCN+GRU
(0.77 MB). Target: 2D fingertip velocity (auto-selected movement axes 1,2).

Results (Pearson r on HELD-OUT sessions):
| Held-out session | r_mean | axis1 | axis2 |
| --- | ---: | ---: | ---: |
| 20161017_02 | 0.864 | 0.835 | 0.892 |
| 20161024_03 | 0.849 | 0.824 | 0.875 |
| mean | 0.856 | | |

Interpretation (KEY): the model generalises to ENTIRELY UNSEEN SESSIONS at
r 0.856 -- essentially the same as within-session (0.848-0.894). So it learns
genuine, session-stable motor encoding, not per-session overfitting. This is a
true held-out generalisation test (test sessions never in the training set), and
it holds up. Model stays 0.77 MB (192,770 params, 96->F). Trained on 6 sessions;
scaling to more training sessions should hold or improve.

Correction to LOG-025: cross-session pooling IS valid with per-electrode
features (not sorted units). Both are legitimate: LOG-025 = within-session
sorted-unit decode; LOG-026 = cross-session per-electrode generalisation.

### LOG-027 - Cross-SUBJECT Test: Same-Day Works, Different-Monkey Collapses

Question: does the model work ACROSS PEOPLE (subjects), or only across days for
the same subject?

Method: `tools/indy_crosssession.py` (TEST = one held-out indy session + one loco
session). Train on 6 indy sessions (per-electrode 96 ch). Test on: (a) a held-out
indy session (same monkey, unseen day), (b) a loco session (the OTHER monkey).
Loco subset to first 96 electrodes to match input size.

Results (Pearson r on held-out sessions):
| Held-out test | r_mean | axis1 | axis2 |
| --- | ---: | ---: | ---: |
| indy 20161017_02 (same subject, unseen day) | 0.864 | 0.835 | 0.892 |
| loco 20170215_02 (DIFFERENT subject) | -0.048 | 0.012 | -0.107 |

Interpretation (KEY, answers "does it work across people"): NO. Same-subject
across-days generalises (0.864); cross-subject collapses to ZERO (-0.048). An
indy-trained model has no predictive power on loco because loco's 96 electrodes
record different neurons in a different brain -- the input channels do not
correspond across subjects. Cross-subject intracortical decoding needs per-person
calibration or neural-alignment ("stitching") methods, not transfer of a trained
model. This is expected and is the fundamental reason intracortical BCIs are
calibrated per person.

Caveat: loco was decoded on indy's fixed velocity axes; even with perfect axes,
the neural input is non-corresponding, so collapse is driven by the electrode/
neuron mismatch, not axis choice. Note: EEG differs -- shared 10-20 scalp
positions make cross-subject at least geometrically possible (untested for
velocity here).

Summary of the generalisation ladder (finger velocity, TCN+GRU 0.77-0.81 MB):
within-session ~0.85-0.89 | across-days same-subject 0.856 | across-subjects ~0.

### LOG-028 - Multi-Band Filter-Bank Input (negative) + Band-Gating Setup

Question: does feeding multiple frequency bands as extra channels (delta movement
potential + mu/beta/low-gamma envelopes) beat the single low band for EEG->finger
velocity? (Literature says mu/beta bandpower also encode kinematics.)

Method: `tools/way_gal_kin_research.py` load_mb (raw low band + Hilbert amplitude
envelopes of rhythm bands, stacked as channels) + BANDSETS presets. Sweep on P1
(--stage mband), then validate best (lp4+mu+beta) on 3 subjects (--stage
final_mband, resumable per subject).

Results:
- P1 band sweep (60 ep): lp2 single band 0.880 BEATS every multi-band variant
  (lp2+mu 0.858, lp2+mu(8-10) 0.855, lp2+mu+beta 0.864, +lowgamma 0.866,
  lp4+mu+beta 0.868).
- 3-subject validation (lp4+mu+beta, 150 ep): P1 0.867, P2 0.778, P3 0.840,
  MEAN 0.828 -- WORSE than single-band best (0.853) on ALL three subjects.

Interpretation: naive multi-band CONCATENATION hurts (-0.025 mean). For real
movement EXECUTION the <2 Hz movement-related potential dominates; adding mu/beta
envelope channels dilutes it and adds trainable input dims that don't help. (mu/
beta ERD matters more for IMAGERY than execution.) Honest negative for concat.

Follow-up (user idea): instead of always concatenating, LEARN a gate that weights
bands adaptively -- "use beta only when it helps." Added a BandGate module to
build_net: 'static' (one weight per band) and 'dynamic' (per-band, per-timestep
gate from a small conv). run_nn can return the learned gate profile (ret_gate) so
we can read out the pattern (which band, when). Stage `--stage gate`. Results in
LOG-029.

### LOG-029 - Learned Band-Gating: No Gain, No Strong Pattern (honest negative)

Question (user idea): instead of always concatenating bands, LEARN a gate that
feeds each band only when it helps -- and see if an interpretable "law/pattern"
emerges (which band, when).

Method: `--stage gate` on P1. Input = lp2+mu+beta (delta + mu/beta envelopes,
3 bands x 32 ch). BandGate module: 'static' (one learned weight per band) and
'dynamic' (per-band, per-timestep gate from a small conv). Read out the learned
gate via run_nn(ret_gate=True).

Results (P1, 3-fold):
- single band lp2 (reference): 0.880
- concat (no gate): 0.864
- band-gate static: 0.864 | learned weights delta 0.514, mu 0.483, beta 0.504
- band-gate dynamic: 0.860 | gates ~0.5 all bands; only faint mu rise post-onset
  (pre 0.491 -> post 0.531)

Interpretation: gating did NOT help (0.86 ~ concat, both < single-band 0.88) and
learned NO strong pattern -- the gate sat near uniform (~0.5), never suppressing
mu/beta. The only directional hint (mu gate up slightly post-movement-onset) is
consistent with mu ERD but negligible in effect. Root cause: mu/beta envelopes
carry little COMPLEMENTARY velocity info beyond the <2 Hz movement potential for
this EXECUTION task, so there is nothing useful for the gate to select -- gating
cannot create signal that is not there. Even the best case (gate learns to drop
mu/beta) would only recover single-band 0.88, not beat it.

The interpretable "law" that emerged is the null one: for movement-EXECUTION
velocity decoding, the delta (<2 Hz) movement-related potential is the workhorse;
mu/beta rhythms add ~nothing (despite their role in motor IMAGERY in the
literature). Model of record stays single-band BIGP (mean r 0.853).

Possible (low-ROI) refinement not pursued: add a sparsity/entropy penalty to push
gates toward 0/1 so they must commit -- but best case only ties single-band.

### LOG-030 - Velocity-Target Low-Pass Sweep on Monkey Data (small real gain)

Question (user): did we ever try a low-pass at 3 Hz? -> No: EEG swept 2/4/8/12
(2 best); the MONKEY pipeline used NO low-pass (raw numerical gradient of finger
position = noisy). Test low-passing the velocity target (arm velocity is
band-limited ~<5 Hz, so this removes derivative/marker jitter).

Method: `tools/indy_vellp.py`. indy_20161005_06, sorted-unit rates (50 ms bins),
low-pass finger position before gradient, sweep cutoff. TCN+GRU, within-session
5-block CV.

Results (r_mean):
| vel-LP cutoff | r |
| --- | ---: |
| none (raw) | 0.850 |
| 8 Hz | 0.850 |
| 6 Hz | 0.850 |
| 4 Hz | 0.853 |
| 3 Hz | 0.856 |
| 2 Hz | 0.860 |

Interpretation: low-passing the velocity target monotonically raises r as the
cutoff drops; >=6 Hz has no effect (velocity has ~no energy there at 20 Hz
sampling). 3 Hz -> +0.006, 2 Hz -> +0.010. Real but modest. CAVEAT: part of the
gain is genuine (removing numerical-derivative/marker noise) and part is the
easier-target effect (smoother signal is easier to predict); pushed too far it
would remove real movement dynamics. Reach dynamics run ~3-5 Hz, so 3-4 Hz is the
honest sweet spot; 2 Hz is borderline over-smoothing. Recommend adopting a 3 Hz
velocity low-pass as the monkey pipeline default and re-checking the cross-session
held-out headline (was 0.856 with no LP).

### LOG-031 - Real-Time Inference Latency (and the causality caveat)

Question: is the velocity decoder fast enough for real-time, and does it work
causally? Real-time control rate = one prediction per 50 ms bin (20 Hz).

Method: time one forward pass on a 2 s window (96 ch x 40 bins) on 1 CPU core.

Results (1 core):
| Model | params | ms/pred | preds/s | margin vs 50 ms |
| --- | ---: | ---: | ---: | ---: |
| bidirectional (best) | 192,770 | 5.80 | 172 | 9x |
| causal unidirectional | 118,146 | 3.71 | 270 | 13x |
| causal small (H32,F32) | 31,426 | 3.06 | 326 | 16x |

Deployment: sliding window [now-2s, now] every 50 ms -> read the last time-step as
current velocity. ~6 ms latency << 50 ms budget -> real-time with big headroom,
single core.

CAVEAT (causality): the best model uses a BIDIRECTIONAL GRU. Over a past window it
is deployable (window holds only past samples; use the last step). But offline CV
scores ALL time-steps, and middle steps see within-window "future" -- at
deployment only the last step (past-only) is available, which may be slightly
less accurate. The causal UNIDIRECTIONAL model avoids this mismatch and is faster
(3.7 ms); its accuracy cost vs bidirectional is measured in LOG-032.

### LOG-032 - Monkey Tuning: Rate Smoothing + Causal (real-time) Cost

Question: does input firing-rate smoothing add on top of the 3 Hz vel-LP, and
how much accuracy does the honest real-time (causal) model cost?

Method: `tools/indy_tune.py`. Session indy_20161005_06, vel-LP 3 Hz fixed. Sweep
Gaussian firing-rate smoothing sigma (bins) with bidirectional GRU; then the best
sigma with a CAUSAL unidirectional GRU. Within-session 5-block CV.

Results (r_mean):
| Config (vel-LP 3 Hz) | r |
| --- | ---: |
| rate-smooth sigma=0 | 0.856 |
| rate-smooth sigma=1 (50 ms) | 0.859 (best) |
| rate-smooth sigma=2 | 0.857 |
| rate-smooth sigma=3 | 0.841 (over-smoothed) |
| sigma=1 CAUSAL unidir (real-time) | 0.854 |

Cumulative (within-session): raw 0.850 -> +3 Hz vel-LP 0.856 -> +sigma1
rate-smooth 0.859 (offline bidirectional). Real-time causal = 0.854 (only -0.005
vs bidirectional).

Interpretation: input rate-smoothing (sigma=1) adds a small clean +0.003 without
touching the target (no over-smoothing concern); sigma>=3 over-smooths the input.
KEY: the causal unidirectional model (the honest real-time decoder, 3.7 ms/pred,
past-only) is only 0.005 below the offline bidirectional best -- real-time
deployment is nearly free here. Best offline config of record for monkey:
per-electrode rates + 3 Hz vel-LP + sigma1 rate-smooth, TCN+GRU; real-time =
causal variant. Next: re-run cross-session held-out with this config.

## Entry Template

### LOG-XXX - Short Name

Question:

Method:

Command:

Artifacts:

Results:

Interpretation:

Decision:
