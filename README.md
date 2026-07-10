# Neural Intent Decoder

This repository decodes continuous 2D hand/finger velocity from intracortical
spike rates with a compact TCN+GRU sequence model.

## Current model

The only active Python package is [`models/`](models/). The readable architecture
is in [`models/best_model.py`](models/best_model.py); `checkpoint.pt` contains
only its learned weights. The package also includes the exact configuration,
held-out evaluation, and training entry point.

| Input setting | Held-out cross-session Pearson r |
| --- | ---: |
| 96 electrodes | **0.87** |
| Top 8 electrodes by firing rate | 0.76 |

```bash
py models/crosssession.py   # reproduce held-out evaluation
py models/train_and_save.py # retrain models/checkpoint.pt
```

The model uses 40 ms bins, a 2 s window, per-electrode multiunit spike rates,
and predicts the two dominant fingertip-velocity axes. It generalizes across
sessions from the same subject, but not across subjects.

## Repository map

```text
models/                  readable best-model source plus learned checkpoint
legacy/                  concluded pipelines and old experiment trials
  monkey_trials/         intracortical sweeps and ablations
  src/ and tools/        earlier EEG/fNIRS and WAY-EEG-GAL work
project_memory/          current summary and chronological daily log
data/                    local datasets (gitignored)
results/                 generated metrics and figures
```

Start with [`project_memory/SUMMARY.md`](project_memory/SUMMARY.md) for the
current research state and [`project_memory/DAILY_LOG.md`](project_memory/DAILY_LOG.md)
for experiment provenance. Legacy code is retained for reproducibility, not as
the recommended entry point.
