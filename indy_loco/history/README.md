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
- A 64-channel comparison improved single-seed December validation to R² 0.6625, but it was not confirmed across seeds and was never promoted.

## Where to read

- `../docs/STATUS.md`: final technical snapshot and validity limits.
- `EXPERIMENT_LOG.md`: concise experiment history and decisions.
- `EXPERIMENT_CODE_INDEX.md`: maps experiment phases to scripts and outputs.
- `experiments/`: archived experiment scripts.
- `results/indy/`: metrics, figures, and intermediate artifacts.

The retained implementation and checkpoints are under `../models/indy_32ch/`.
Data and processing code are under `../data/`.

## Archive rule

Do not modify these artifacts to support new work. If Indy Loco work is resumed,
create a runner under `../experiments/active/` and re-establish the evaluation
protocol explicitly.
