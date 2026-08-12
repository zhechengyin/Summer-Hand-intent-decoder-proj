# Embedded Neural Signal Models

This repository develops compact neural-signal models intended for eventual
low-latency firmware deployment.

The current task is FingerMovements left/right EEG classification from 28
channels and 50 samples at 100 Hz. The active model is the frozen strictly
causal Phase 2c CSSD + hierarchical LDA, selected entirely from corrected
official MATLAB TRAIN data.

## Current result

| Model | Mean OOF balanced accuracy |
|---|---:|
| Terminal features + Logistic reproduction | 78.58% |
| Paper-style CSSD + hierarchical LDA | 85.03% |
| Phase 2b zero-phase offline reference | 86.72% |
| Phase 2c causal horizon diagnostic at 500 ms | 82.93% |
| Phase 2c causal 500 ms / 50 ms baseline | 82.93% |
| **Phase 2c causal 400 ms / 50 ms model** | **83.99%** |
| Phase 2d retrospective official TEST of Phase 2c model | 77.05% |
| Phase 2e regularized CSSD | 84.10% |
| Phase 2e ToeplitzLDA exploratory result | 84.50% |
| Phase 2e baseline + Toeplitz nested fusion | 84.09% |
| Phase 2f low-dimensional Riemannian candidate | 85.13% |

The active causal configuration uses empirical covariance, per-trial trace
normalization, one BP and one ERD/F2 spatial pattern per class, and LDA fusion.
Its seed standard deviation was 0.54 percentage points and its worst-seed OOF
balanced accuracy was 83.25% across seeds 42/43/44.

The causal implementation and verified all-TRAIN checkpoint are under
`models/finger_movements/cssd_lda/`. All completed experiment code and results,
including the Phase 2d official-TEST evaluation and the Phase 2e/2f model
comparisons, are archived under `history/finger_movements/`. Phase 2f improved
mean OOF BA to 85.13% and worst-seed BA to 84.17%, but seed and fold variability
increased and seed 43 did not improve. It therefore failed the predeclared
promotion rule. The Phase 2c checkpoint remains frozen, model exploration is
closed, and the next work is firmware deployment validation.

The frozen 400 ms checkpoint achieved 77.05% balanced accuracy on the 100-case
corrected official TEST. This is a retrospective benchmark rather than a
pristine blind test because TEST had been exposed earlier in the project.

## Important limitation

The former 86.72% Phase 2b checkpoint uses zero-phase temporal filtering and is
archived as an offline reference. The active Phase 2c model reaches 83.99%
TRAIN-only mean OOF balanced accuracy with strict left-to-right filtering and
a 400 ms feature ring ending at the current prediction point. A cold reset
uses 100 ms filter pre-roll, so first output remains at 500 ms; steady-state
updates are every 50 ms. Continuous EEG and rest/no-intent validation remain
required before firmware deployment.

## Project rules

1. Every model has one explicit input contract and one explicit target.
2. Learned preprocessing is fitted from training folds only.
3. Active code does not import from `history/`.
4. Results produced from the retired UEA sliding-channel conversion are
   historical provenance only and are not comparable with corrected results.
5. The official TEST has already been exposed and must not be used for model
   selection; a new external holdout is needed for independent final evidence.

## Repository layout

```text
data/raw/FingerMovements/               immutable official source files
data/processed/finger_movements/        model-ready official splits
data/processing/finger_movements/       supported conversion code
models/finger_movements/cssd_lda/       active causal model and checkpoint
experiments/active/                     no active experiment
results/                                no active result set
docs/STATUS.md                          current source of truth
history/finger_movements/               completed EEG experiments/results through Phase 2f
history/indy/                           completed Indy project
```
