# Embedded Neural Signal Models

This repository is being restarted around a deployment-first goal: train a
small set of simple models for separate neural-signal prediction or
classification tasks, then measure whether each model is accurate and cheap
enough to run at low latency on firmware.

There is currently no active dataset, model, checkpoint, or experiment. The
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

Define the first new task before implementing a model:

- firmware input signal and sampling rate;
- prediction or classification label;
- dataset and split policy;
- latency, RAM, Flash, and minimum-performance targets.

After that, establish simple baselines before considering a larger
architecture.
