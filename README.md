# Embedded Neural Signal Models

This repository develops compact neural-signal models intended for eventual
low-latency firmware deployment.

The current task is FingerMovements left/right EEG classification from 28
channels and 50 samples at 100 Hz. The active offline research model is the
Phase 2b CSSD + hierarchical LDA winner, selected entirely from corrected
official MATLAB TRAIN data.

## Current result

| Model | Mean OOF balanced accuracy |
|---|---:|
| Terminal features + Logistic reproduction | 78.58% |
| Paper-style CSSD + hierarchical LDA | 85.03% |
| **Selected Phase 2b CSSD + hierarchical LDA** | **86.72%** |

The selected configuration uses empirical covariance, per-trial trace
normalization, one BP and one ERD/F2 spatial pattern per class, and LDA fusion.
Its seed standard deviation was 0.68 percentage points and its worst-seed OOF
balanced accuracy was 86.09% across seeds 42/43/44.

The active implementation and verified all-TRAIN checkpoint are under
`models/finger_movements/cssd_lda/`. All completed experimental code and
results are archived under `history/finger_movements/`.

## Important limitation

The current model uses zero-phase temporal filtering. It is therefore an
offline reference, not yet a causal streaming firmware model. The next model
phase must replace that preprocessing with a causal implementation and repeat
TRAIN-only validation before deployment.

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
models/finger_movements/cssd_lda/       active model and checkpoint
experiments/active/                     empty experiment boundary + README
results/                                no active experiment results
docs/STATUS.md                          current source of truth
history/finger_movements/               completed EEG experiments/results
history/indy/                           completed Indy project
```
