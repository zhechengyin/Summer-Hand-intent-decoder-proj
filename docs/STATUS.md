# Project Status

Updated: 2026-08-13

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
`models/finger_movements/cssd_lda/`. Each prediction at point A uses a 400 ms
feature ring ending at A, with causal filter state carried across 50 ms
updates. A cold start reserves 100 ms for causal filter pre-roll before the
400 ms ring. The Phase 2b zero-phase and Phase 2c 500 ms models are archived.
Phase 2e completed its TRAIN-only paired comparison of regularized CSSD,
shrinkage LDA, their combination, and block-Toeplitz LDA against the unchanged
Phase 2c baseline. None met the predeclared stability criteria, so no model or
checkpoint was promoted. Phase 2f then tested a low-dimensional Riemannian
tangent-space candidate on the same seeds/folds. It improved mean and
worst-seed BA, but seed/fold variability increased and seed 43 did not improve,
so it failed the frozen promotion rule. No checkpoint changed. Phase 2d/2e/2f
evidence is archived, model exploration is closed, and firmware deployment
validation is next. A self-contained float32 C99 port has been added under the
active model without changing its Python implementation or checkpoint.

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
| **Promoted Phase 2c causal model at 400 ms** | **83.99%** | **0.54 pp** | **83.25%** |
| Phase 2e regularized CSSD | 84.10% | 0.60 pp | 83.25% |
| Phase 2e ToeplitzLDA | 84.50% | 0.90 pp | 83.23% |
| Phase 2e baseline + Toeplitz nested fusion | 84.09% | 0.98 pp | 83.24% |
| Phase 2f low-dimensional Riemannian | 85.13% | 1.14 pp | 84.17% |

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
models/finger_movements/cssd_lda/checkpoints/finger_movements_cssd_lda_phase2c_causal_400ms.npz
```

Checkpoint SHA-256:

```text
87b84cc2c8baf9efdc1ccf37ad28f5f58ad13c4db2a8f8a273fe73fce9956101
```

The causal checkpoint was fitted once on all 316 official TRAIN cases, saved,
reloaded, and verified with identical predictions and zero reload error.
Processing each case as ten stateful 50 ms chunks (100 ms filter pre-roll plus
the 400 ms ring) reproduced batch inference with maximum score error
`3.20e-14` and probability error `2.11e-15`. Apparent fitting balanced accuracy
was 89.89%; this is a fit diagnostic, not a held-out estimate. Official TEST
was refused while this checkpoint was selected and fitted, then opened later
only by the frozen Phase 2d inference runner.

## Deployment boundary

The active checkpoint uses fourth-order causal Butterworth SOS filtering and
has a streaming state interface. However, the dataset provides only isolated
500 ms epochs. Continuous EEG is still needed to verify long-running filter
state, overlapping predictions, and behavior between labeled events. The
dataset also lacks a rest/no-intent class, and a new external holdout is still
required for a genuinely independent final claim.

## Completed Phase 2c bin/window sweep

The archived runner is:

```text
history/finger_movements/experiments/phase2c_bin_window_sweep.py
```

It compared past-context windows of 200/300/400/500 ms under the same
TRAIN-only seeds 42/43/44 and five folds. Mean OOF BA was 79.62%, 79.43%,
83.99%, and 82.93%, respectively. The frozen 400 ms winner improved mean BA by
1.05 points, worst-seed BA by 1.57 points, and seed SD by 0.49 points over
500 ms. Its seed BAs were 83.25%, 84.20%, and 84.52%.

Streaming bins of 10/20/50/100 ms all reproduced the same filtered endpoint
with exactly zero numerical error. Bin size is an update-cadence/firmware
choice, not an accuracy hyperparameter for this feature model. The runner and
all result files are archived; no experiment remains active.

## Phase 2d retrospective official TEST

Phase 2d applied the frozen Phase 2c 400 ms checkpoint once to the corrected
100-case official TEST through pure inference. No fitting, recalibration,
threshold selection, or TEST-derived preprocessing occurred. Phase 2d is an
evaluation record only and does not rename or modify the Phase 2c model.

| Metric | Result |
|---|---:|
| Accuracy | 77.00% |
| Balanced accuracy | 77.05% |
| Macro-F1 | 77.00% |
| Left recall | 79.59% |
| Right recall | 74.51% |
| Accuracy Wilson 95% CI | 67.85%--84.16% |

The confusion matrix was `[[39, 10], [13, 38]]` in left/right order. TEST BA
was 6.94 points below the frozen 83.99% TRAIN-only OOF estimate. Batch and ten
stateful 50 ms-chunk predictions were identical, with maximum score error
`9.77e-15` and probability error `1.78e-15`.

This is a retrospective benchmark, not a pristine blind test, because official
TEST was exposed earlier in the project. The checkpoint and model selection
must not be changed in response to this result.

Archived evidence:

```text
history/finger_movements/experiments/phase2d_evaluate_frozen_test.py
history/finger_movements/results/phase2d_official_test_400ms/
```

## Completed Phase 2e lightweight comparison

Phase 2e used the exact Phase 2c seeds 42/43/44, five deterministic stratified
folds per seed, 400 ms causal feature ring, and 50 ms update contract. Every
learned covariance, CSSD projection, scaler, and classifier was fitted inside
its current training fold. Official TEST was refused.

| Variant | Mean OOF BA | Seed SD | Worst seed | Fold SD | Seed BA deltas vs baseline |
|---|---:|---:|---:|---:|---|
| Current CSSD + LDA | 83.99% | 0.54 pp | 83.25% | 3.91 pp | reference |
| Regularized CSSD | 84.10% | 0.60 pp | 83.25% | 4.24 pp | +0.01 / +0.32 / 0.00 pp |
| Shrinkage LDA | 83.66% | 0.54 pp | 82.92% | 4.96 pp | +0.94 / -0.33 / -1.59 pp |
| Regularized CSSD + shrinkage LDA | 83.55% | 0.45 pp | 82.91% | 4.75 pp | +0.62 / -0.33 / -1.60 pp |
| ToeplitzLDA | 84.50% | 0.90 pp | 83.23% | 4.19 pp | +1.89 / +0.94 / -1.28 pp |
| Baseline + Toeplitz nested fusion | 84.09% | 0.98 pp | 83.24% | 4.57 pp | 0.00 / +1.27 / -0.96 pp |

ToeplitzLDA corrected 28.3%, 30.0%, and 18.4% of baseline errors in the three
seeds, so it passed the predeclared complementarity gate. A four-fold
inner-OOF shrinkage-LDA stacker was therefore evaluated in each outer fold.
The fusion did not stabilize the gain: it improved only seed 43, increased
seed/fold variability, and added parameters.

All 5,688 OOF float32 predictions exactly matched the float64 reference. Each
single model retained the same estimated deployment footprint as the baseline:
323 float parameters, 1.28 KB parameter storage, and 12.19 KB working RAM. The
fusion required 1.99 KB of parameters.

Decision: retain the frozen Phase 2c empirical CSSD + SVD-LDA checkpoint.
ToeplitzLDA is recorded as an exploratory mean-BA improvement, not a promoted
model. The metrics JSON field selecting Toeplitz by mean-first ordering is a
mechanical ranking only and does not override this multi-criterion decision.

Evidence:

```text
history/finger_movements/experiments/phase2e_lightweight_regularization_comparison.py
history/finger_movements/results/phase2e_lightweight_comparison/
```

## Completed Phase 2f Riemannian comparison

Phase 2f compared the unchanged Phase 2c baseline with one self-contained
low-dimensional Riemannian candidate on the exact seeds 42/43/44, five folds,
400 ms causal ring, and 50 ms update contract. Official TEST was refused, and
all learned operations were fit inside each training fold.

| Variant | Mean OOF BA | Seed SD | Worst seed | Fold SD |
|---|---:|---:|---:|---:|
| Phase 2c baseline | 83.99% | 0.54 pp | 83.25% | 3.91 pp |
| Low-dimensional Riemannian | 85.13% | 1.14 pp | 84.17% | 4.79 pp |

The Riemannian seed deltas were +1.25, -0.03, and +2.21 percentage points.
Mean BA and worst-seed BA improved, and every float32 OOF label matched the
float64 reference. However, seed 43 did not improve and both seed and fold
variability increased. The candidate therefore failed two of five predeclared
promotion requirements. Its 479 float parameters require about 1.89 KB,
estimated working RAM is 14.02 KB, and inference adds two 4x4 matrix
eigendecompositions per update.

Decision: retain the frozen Phase 2c checkpoint, promote no Phase 2f model,
stop further model-family exploration, and proceed to firmware deployment
validation. Archived evidence:

```text
history/finger_movements/experiments/phase2f_low_dimensional_riemannian.py
history/finger_movements/results/phase2f_riemannian/
```

## Firmware C port

The active frozen Phase 2c checkpoint is exported into a portable C99
implementation at:

```text
models/finger_movements/cssd_lda/firmware/
```

The port contains no fitting code, dynamic allocation, Python dependency, or
runtime file loading. It implements the causal BP/ERD SOS filters, persistent
400 ms circular buffers, frozen CSSD projections, BP/ERD/trend features, three
branch LDAs, and final LDA fusion using float32. The LDA normalization and
coefficients are algebraically folded into equivalent inference weights.

The exporter verifies checkpoint SHA-256
`87b84cc2c8baf9efdc1ccf37ad28f5f58ad13c4db2a8f8a273fe73fce9956101`
before generating C constants. Host C validation over all 316 official TRAIN
cases produced zero class-label mismatches against Python and zero mismatches
between one 500 ms block and ten 50 ms chunks. Maximum absolute score and
probability errors were `1.335e-4` and `3.228e-5`.

Persistent state is 10,312 bytes (10.07 KiB); core numeric constants plus trend
indices occupy 903 bytes before alignment/metadata. This is a source-level
firmware candidate, not yet a target-board qualification. Remaining work is to
integrate the selected STM32 acquisition path, confirm EEG scaling/channel
order, measure Flash/RAM/stack/cycles on the real toolchain, and validate
continuous EEG including rest/no-intent behavior.

## Repository collaboration tooling

The repository now provides a root `Makefile`, `.pre-commit-config.yaml`,
`ruff.toml`, `pytest.ini`, and separate `requirements-dev.txt`. `make
setup-dev` installs runtime/development dependencies plus pre-commit and
pre-push hooks. `make lint`, `make format`, `make test`, and `make
firmware-test` provide the standard local workflow.

Two initial pytest guardrails protect the frozen checkpoint SHA-256 and require
the generated C parameters to identify that same checkpoint. Ruff and generic
pre-commit checks apply to active code/configuration; immutable `history/` and
`data/raw/` are excluded from mechanical rewrites. `make data-archive` shares
only `data/processed/`, not raw source data. Full contribution and experiment
governance is documented in `CONTRIBUTING.md`.
