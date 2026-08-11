# FingerMovements processed data

This directory contains a direct, model-ready conversion of the official
BCI Competition II Data Set IV 100 Hz MATLAB release.

## Dataset

- Task: binary classification of left- versus right-hand finger movement.
- Signal: 28-channel scalp EEG.
- Sampling rate: 100 Hz after the dataset authors' downsampling.
- Case duration: 500 ms, represented by 50 timepoints at 10 ms intervals.
- Timing: each case ends approximately 130 ms before the key press.
- Official split: 316 training cases and 100 test cases. The test was opened
  for the frozen Phase 1h checkpoint on 2026-08-05 and for pure inference of
  the frozen Phase 2c 400 ms checkpoint during Phase 2d on 2026-08-11.
- Labels: `left = 0`, `right = 1`.

The release describes three same-day recording sessions but does not provide a
session identifier for each epoch. Do not infer session boundaries from
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

The converter reads `sp1s_aa.mat` and the separately published official
`labels_data_set_iv.txt`. It transposes EEG from the official
`time x channels x trials` layout to `trials x channels x time` and rejects the
deterministic adjacent-channel overlap found in the retired UEA conversion.

No normalization, filtering, feature extraction, augmentation, shuffling, or
split reassignment is applied. Normalization parameters must be learned from a
training subset only. The official test was opened once after the Phase 1h
model and hyperparameters were frozen. Future model selection must not use its
labels or metrics; use the 316-case training split for cross-validation.

The raw source files under `data/raw/` are never modified by the converter.

## Rebuild

From the repository root, run:

```bash
python data/processing/finger_movements/prepare_finger_movements.py
```

The conversion script and this README are the source of truth for the schema;
there is intentionally no separate `manifest.json`.
