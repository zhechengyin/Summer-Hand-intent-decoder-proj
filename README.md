# Embedded Neural Signal Models

This repository is being restarted around a deployment-first goal: train a
small set of simple models for separate neural-signal prediction or
classification tasks, then measure whether each model is accurate and cheap
enough to run at low latency on firmware.

FingerMovements is now the active dataset, and Phase 1b is the first active
model-family comparison. There is not yet a selected model or checkpoint. The
completed Indy Loco program has been preserved under
[`history/indy/`](history/indy/README.md) and is no longer an active dependency.

## Project rules

1. Each model must have one explicit input contract and one explicit target.
2. Datasets with different label meanings are not merged into one supervised
   task merely because they share a signal modality.
3. Train, validation, and test boundaries are defined before tuning.
4. Accuracy and firmware cost are evaluated together: parameters, peak RAM,
   Flash, latency, and numerical equivalence all matter.
5. Active code must not import from `history/`.

## Active repository layout

```text
configs/             active dataset and model configurations
data/
  raw/               immutable source datasets
  processed/         generated model-ready data
  processing/        supported conversion and inspection code
models/              one self-contained directory per active model
experiments/active/  current controlled experiments only
results/             results grouped by task and experiment
docs/
  STATUS.md          current project truth and next gate
  history/
    EXPERIMENT_LOG.md
history/
  indy/              complete read-only Indy project archive
```

## Next gate

Run the registered Phase 1b experiment on the official FingerMovements training
split. Compare repeated cross-validation accuracy, seed stability, and parameter
count for the four lightweight baselines. Keep the official 100-case test split
locked until a model family and training policy have been frozen.
