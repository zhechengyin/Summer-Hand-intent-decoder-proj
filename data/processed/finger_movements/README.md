# FingerMovements processed data

This directory contains a direct, model-ready conversion of the official
FingerMovements split from the UEA multivariate time-series archive.

## Dataset

- Task: binary classification of left- versus right-hand finger movement.
- Signal: 28-channel scalp EEG.
- Sampling rate: 100 Hz after the dataset authors' downsampling.
- Case duration: 500 ms, represented by 50 timepoints at 10 ms intervals.
- Timing: each case ends approximately 130 ms before the key press.
- Official split: 316 training cases and 100 locked test cases.
- Labels: `left = 0`, `right = 1`.

The archive does not retain subject, recording-session, trial-time, or
train/validation grouping metadata. Do not infer those boundaries from
`source_index`.

## Files and schema

`train.npz` contains the 316 official training cases. `test.npz` contains the
100 official test cases. Both files contain exactly four arrays:

| Array | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `x` | `float32` | `(N, 28, 50)` | EEG ordered as case, channel, timepoint |
| `y` | `uint8` | `(N,)` | Class ID: left 0, right 1 |
| `source_index` | `int32` | `(N,)` | Original zero-based row within that source split |
| `channel_names` | Unicode | `(28,)` | EEG channel names in `x` channel order |

Channel order:

```text
F3, F1, Fz, F2, F4, FC5, FC3, FC1, FCz, FC2, FC4, FC6,
C5, C3, C1, Cz, C2, C4, C6, CP5, CP3, CP1, CPz, CP2,
CP4, CP6, O1, O2
```

Class counts:

| Split | Left | Right | Total |
| --- | ---: | ---: | ---: |
| Train | 159 | 157 | 316 |
| Test | 49 | 51 | 100 |

## Processing policy

The converter reads only `FingerMovements_TRAIN.ts` and
`FingerMovements_TEST.ts`. The ARFF files in the downloaded archive are
redundant representations and are not inputs to this conversion.

No normalization, filtering, feature extraction, augmentation, shuffling, or
split reassignment is applied. Normalization parameters must later be learned
from a training subset only. The official test file must remain locked until
the model and hyperparameters are frozen.

The raw source files under `data/raw/` are never modified.

## Rebuild

From the repository root, run:

```bash
python data/processing/finger_movements/prepare_finger_movements.py
```

The conversion script and this README are the source of truth for the schema;
there is intentionally no separate `manifest.json`.
