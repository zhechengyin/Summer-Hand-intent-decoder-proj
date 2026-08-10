# Project Status

Updated: 2026-08-10

## Current state

FingerMovements Phase 2c now has a promoted **strictly causal candidate**:

```text
empirical CSSD covariance
per-trial trace normalization enabled
one BP spatial pattern per class
one ERD/F2 spatial pattern per class
BP + ERD + BP-trend branch LDAs
final LDA fusion
```

The implementation and verified all-TRAIN checkpoint are under
`models/finger_movements/cssd_lda/`. Each prediction at point A uses only
`[A-500 ms, A]`, with causal filter state carried across 50 ms updates. The
Phase 2b zero-phase model is now an archived offline reference. The active
experiment remains Phase 2c and sweeps history-window and streaming-bin sizes.

## Valid data contract

- source: official BCI Competition II Data Set IV MATLAB release;
- task: binary left/right movement classification;
- input: 28 EEG channels, 50 samples at 100 Hz per case;
- development split: 316 official TRAIN cases, left 159/right 157;
- processed TRAIN SHA-256:
  `a2025f277b5351839554e0ecf3398f1f4fd5151a4fc90f0e25c873734f5a91d1`;
- official TEST: previously opened during the invalid UEA-conversion phase and
  unavailable as a pristine selection gate.

The former UEA conversion had deterministic adjacent-channel overlap and is
retired. Its Phase 1, original Phase A2, isolated Phase 2b, and official-test
scores remain in history only as provenance; they are not valid evidence for
the corrected dataset.

## Corrected TRAIN-only evidence

All figures below use seeds 42/43/44 with five stratified folds per seed. Every
learned spatial filter, scaler, and classifier was fitted within its training
fold. Official TEST was not loaded.

| Model | Mean OOF BA | Seed SD | Worst seed |
|---|---:|---:|---:|
| Archived terminal features + Logistic, re-evaluated | 78.58% | 1.04 pp | 77.22% |
| Paper-style Phase A2 CSSD + hierarchical LDA | 85.03% | 1.27 pp | 83.25% |
| **Promoted Phase 2b configuration** | **86.72%** | **0.68 pp** | **86.09%** |
| Phase 2c causal model at 500 ms | 82.93% | 1.03 pp | 81.67% |

The Phase 2b winner improved all three seeds over the corrected Phase A2
reference. It was empirical covariance, trial trace normalization on, one F2
component per class, and LDA fusion. This supersedes the old invalid-data
conclusion that trial normalization should be off.

## Phase 2c causal horizon diagnostic

The initial Phase 2c diagnostic uses five samples per 50 ms bin and evaluates
predictions at ten horizons from 50 through 500 ms. Temporal filtering is
strictly left-to-right
`sosfilt`; each horizon's CSSD filters and LDAs are fitted only from
outer-training trials and only from samples available by that horizon. Whole
trials remain the fold unit.

Mean OOF balanced accuracy versus observation time was:

| Horizon | Mean OOF BA |
|---:|---:|
| 50 ms | 50.62% |
| 100 ms | 51.27% |
| 150 ms | 55.16% |
| 200 ms | 56.33% |
| 250 ms | 62.97% |
| 300 ms | 68.97% |
| 350 ms | 72.89% |
| 400 ms | 77.22% |
| 450 ms | 79.65% |
| 500 ms | 82.93% |

At 500 ms the causal cost versus the frozen zero-phase baseline is 3.79
percentage points. The per-seed causal results were 84.20%, 82.92%, and 81.67%
for seeds 42, 43, and 44. Replacing every future sample after each prediction
horizon with extreme noise changed current scores and probabilities by exactly
zero; official TEST was refused.

The result established the causal candidate that has now been fitted once on
all official TRAIN cases and promoted without touching official TEST.

## Phase 2c past-only streaming result

The final Phase 2c runner corrects the timing interpretation of the initial
Phase 2c horizon endpoint. The 500 ms interval is historical context ending at
the current prediction point A; the model never waits for or consumes samples
after A. Data enters in
50 ms bins, causal filter state is carried between the ten startup bins, and a
500 ms filtered ring buffer supplies endpoint features. After the one-time
startup warm-up, the design can update every 50 ms.

The TRAIN-only endpoint result exactly reproduced the initial Phase 2c horizon
diagnostic at 82.93% mean OOF balanced accuracy, 1.03-point seed SD, and
81.67% worst-seed BA. This is
expected: Phase 2c changes the streaming implementation and timing semantics,
not the endpoint feature definition.

Two executable safeguards ran on all 64 held-out cases in the first fold:

- ten stateful 50 ms filter calls exactly reproduced one full causal pass;
- the rolling buffer emitted first after ten startup bins and again after only
  one additional 50 ms bin;
- adding ten extreme future samples after A changed filtered history, scores,
  probabilities, and predictions at A by exactly zero.

The official dataset contains isolated 500 ms epochs rather than continuous
EEG. Therefore Phase 2c validates the endpoint classifier and bin-wise state
handling inside one epoch, but it cannot validate persistent filter state or
successive overlapping predictions in a real continuous recording.

## Active causal model and checkpoint

Implementation:

```text
models/finger_movements/cssd_lda/model.py
```

Checkpoint:

```text
models/finger_movements/cssd_lda/checkpoints/finger_movements_cssd_lda_phase2c_causal.npz
```

Checkpoint SHA-256:

```text
d92c23f7e6f8722d568d1b31963eab1328d5367ba32764b676d1ae0d73aaefd4
```

The causal checkpoint was fitted once on all 316 official TRAIN cases, saved,
reloaded, and verified with identical predictions and zero reload error.
Processing each case as ten stateful 50 ms chunks reproduced batch endpoint
inference with maximum score error `7.11e-15` and probability error `1.11e-15`.
Apparent fitting balanced accuracy was 88.30%; this is a fit diagnostic, not a
held-out estimate. Official TEST was refused and not loaded.

## Deployment boundary

The active checkpoint uses fourth-order causal Butterworth SOS filtering and
has a streaming state interface. However, the dataset provides only isolated
500 ms epochs. Continuous EEG is still needed to verify long-running filter
state, overlapping predictions, and behavior between labeled events. The
dataset also lacks a rest/no-intent class, and a new external holdout is still
required for a genuinely independent final claim.

## Active Phase 2c bin/window sweep

The active runner is:

```text
experiments/active/phase2c_bin_window_sweep.py
```

It compared past-context windows of 200/300/400/500 ms under the same
TRAIN-only seeds 42/43/44 and five folds. Mean OOF BA was 80.17%, 82.81%,
83.45%, and 82.93%, respectively. The provisional 400 ms winner improved mean
BA by 0.52 points and worst-seed BA by 0.31 points over 500 ms, but seed 42
decreased while seeds 43 and 44 improved. The frozen 500 ms checkpoint is
therefore unchanged pending confirmation.

Streaming bins of 10/20/50/100 ms all reproduced the same filtered endpoint
with exactly zero numerical error. Bin size is an update-cadence/firmware
choice, not an accuracy hyperparameter for this feature model.
