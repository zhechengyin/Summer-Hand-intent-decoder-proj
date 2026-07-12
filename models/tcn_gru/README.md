# Best model

This is the model of record: per-electrode multiunit spike rates to 2D fingertip
velocity using a dilated causal TCN, bidirectional GRU, and per-timestep linear
head.

## Contents

- [`best_model.py`](best_model.py): readable TCN+GRU architecture source and
  training primitives (`build_net` is the model constructor).
- `config.py`: exact preprocessing, model configuration, and recorded metrics.
- `evaluate.py`: fixed file-level train/eval/test pipeline. Validation selects
  the best epoch and test is read only for final scoring.
- `train_and_save.py`: trains the recorded configuration and writes the checkpoint.
- `checkpoint.pt`: learned weights plus config, axes, and target normalization;
  it is not the model source.

## Results

| Setting | Held-out mean Pearson r |
| --- | ---: |
| 96 electrodes | **0.87** |
| Top 8 by training-set firing rate | 0.76 |

The full model is about 192k parameters / 0.77 MB fp32. The configuration uses
40 ms bins, 2 s windows, 3 Hz position low-pass before differentiation, sigma=1
rate smoothing, ReLU TCN blocks, and two output movement axes.

```bash
py models/tcn_gru/evaluate.py
py models/tcn_gru/train_and_save.py
```

All split files are recordings from one subject. Cross-subject transfer failed,
so a new subject requires calibration or alignment. The displayed 0.87/0.76
figures are historical benchmarks and must not be relabeled as results from the
new split until `evaluate.py` is run.
