# Indy Loco Experiment Archive

This directory preserves completed Indy Loco experiments and their evidence.
Nothing here is imported by the retained Indy model or by the independent
FingerMovements project.

## Final outcome

- Task: predict two-dimensional fingertip velocity from binned intracortical spike counts.
- Data: 37 sessions; 29 training, 4 December validation, and 4 January test sessions.
- Final deployable candidate: 32-channel, 48/48 TCN+GRU, 45,266 parameters.
- Final candidate validation: pooled R² 0.5651; session-macro R² 0.5750; worst session R² 0.3461.
- January test had already been inspected: pooled R² 0.5511; one session (`indy_20170124_01`) failed with R² -0.0524. It is historical evidence, not an untouched test set.
- Phase 5 confirmed a 64-channel 64/64 baseline over seeds 42–44: pooled December validation R² `0.6575 ± 0.0080`. Detector-based session removal did not improve the mean.
- Phase 6 promoted a 96-channel 64/64 model with 0.20 paired channel dropout: pooled December validation R² `0.7004 ± 0.0019` over seeds 42–44.
- Phase 7 completed 30 session-local benchmark folds: test R² `0.7056 ± 0.0722`.

## Where to read

- `../docs/STATUS.md`: final technical snapshot and validity limits.
- `EXPERIMENT_LOG.md`: concise experiment history and decisions.
- `EXPERIMENT_CODE_INDEX.md`: maps experiment phases to scripts and outputs.
- `experiments/`: archived experiment scripts.
- `results/indy/`: metrics, figures, and intermediate artifacts.

Retained model implementations and promoted checkpoints are under `../models/`.
Data and processing code are under `../data/`. Completed Phase 6 and Phase 7
runners are under `experiments/phase6/` and `experiments/phase7/`; there is no
active experiment.

## Archive rule

Do not modify these artifacts to support new work. New work belongs under
`../experiments/active/` with its evaluation protocol stated explicitly.
