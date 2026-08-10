# Embedded Neural Signal Models

This repository develops compact neural-signal models intended for eventual
low-latency firmware deployment.

The current task is FingerMovements left/right EEG classification from 28
channels and 50 samples at 100 Hz. The active model is the strictly causal
Phase 2c CSSD + hierarchical LDA candidate, selected entirely from corrected
official MATLAB TRAIN data.

## Current result

| Model | Mean OOF balanced accuracy |
|---|---:|
| Terminal features + Logistic reproduction | 78.58% |
| Paper-style CSSD + hierarchical LDA | 85.03% |
| Phase 2b zero-phase offline reference | 86.72% |
| Phase 2c causal horizon diagnostic at 500 ms | 82.93% |
| **Phase 2c causal 500 ms / 50 ms candidate** | **82.93%** |
| Phase 2c provisional 400 ms window | 83.45% |

The active causal configuration uses empirical covariance, per-trial trace
normalization, one BP and one ERD/F2 spatial pattern per class, and LDA fusion.
Its seed standard deviation was 1.03 percentage points and its worst-seed OOF
balanced accuracy was 81.67% across seeds 42/43/44.

The causal implementation and verified all-TRAIN checkpoint are under
`models/finger_movements/cssd_lda/`. The bin/window sweep is the active Phase
2c experiment under `experiments/active/`; completed runners, results, and the
Phase 2b zero-phase reference are under `history/finger_movements/`.

## Important limitation

The former 86.72% Phase 2b checkpoint uses zero-phase temporal filtering and is
archived as an offline reference. The active Phase 2c model reaches 82.93%
TRAIN-only mean OOF balanced accuracy with strict left-to-right filtering and
uses the 500 ms before the current prediction point. Its all-TRAIN checkpoint
has been promoted, but continuous EEG and rest/no-intent validation are still
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
experiments/active/                     Phase 2c bin/window sweep
results/finger_movements/phase2c_bin_window_sweep/ current sweep evidence
docs/STATUS.md                          current source of truth
history/finger_movements/               completed EEG experiments/results
history/indy/                           completed Indy project
```
