# Neural Intent Decoder

This repository decodes continuous 2D hand/finger velocity from intracortical
spike rates with a compact TCN+GRU sequence model.

## Current model

Model families live under [`models/`](models/). The current model of record is
[`models/tcn_gru/`](models/tcn_gru/), with readable architecture source in
[`models/tcn_gru/best_model.py`](models/tcn_gru/best_model.py); `checkpoint.pt` contains
only its learned weights. The package also includes the exact configuration,
file-level train/eval/test evaluation, and training entry point.

| Historical benchmark | Pearson r |
| --- | ---: |
| 96 electrodes | **0.87** |
| Top 8 electrodes by firing rate | 0.76 |

These numbers predate the new fixed train/eval/test allocation. Run
`models/tcn_gru/evaluate.py` to produce metrics for the new split.

```bash
py models/tcn_gru/evaluate.py       # train/eval/test evaluation
py models/tcn_gru/train_and_save.py # retrain the TCN+GRU checkpoint
```

The eight same-subject `.mat` sessions use the nearest possible whole-file split
to 70/15/15: six training files, one validation file, and one test file
(75/12.5/12.5), named `train1`…`train6`, `eval1`, and `test1`. The model uses
40 ms bins, a 2 s window, and predicts the two dominant velocity axes.

## Repository map

```text
models/                  one self-contained folder per model family
legacy/                  concluded pipelines and old experiment trials
  monkey_trials/         intracortical sweeps and ablations
  src/ and tools/        earlier EEG/fNIRS and WAY-EEG-GAL work
project_memory/          current summary and chronological daily log
data/                    immutable source data plus packaged preprocessing methods
results/                 generated metrics and figures
```

Start with [`project_memory/SUMMARY.md`](project_memory/SUMMARY.md) for the
current research state and [`project_memory/DAILY_LOG.md`](project_memory/DAILY_LOG.md)
for experiment provenance. Legacy code is retained for reproducibility, not as
the recommended entry point.
