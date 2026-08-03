# Project Status

Updated: 2026-08-03

## Active task

FingerMovements binary left/right classification is active. Each case contains
28 EEG channels and 50 samples at 100 Hz. The processed data preserves the
official 316-case train and 100-case test split.

## Frozen model

Feature + Linear is the sole active pipeline:

- seven deterministic features per channel, producing 196 inputs;
- training-derived channel and feature normalization;
- dropout-regularized two-class linear output;
- seed 42, 50 epochs, AdamW;
- learning rate 0.001, weight decay 0.0001, dropout 0.25, batch size 32.

Phase 1c selected it at 60.05% mean OOF balanced accuracy across seeds 42, 43,
and 44. Seed standard deviation was 0.36 percentage points and worst-seed
balanced accuracy was 59.84%. Tiny EEGNet at the same duration reached 59.18%
with 3.01-point seed deviation and a 56.03% worst-seed result.

The comparison is an engineering selection; paired differences were not
statistically significant. The official test was never loaded during model
selection.

## Artifact state

- Final all-training-data checkpoint: not yet trained.
- Official test evaluation: locked and not run.
- Active experiments: none.
- Completed FingerMovements experiments and results:
  `history/finger_movements/`.
- Completed Indy project: `history/indy/`.

## Next gate

Run `models/finger_movements/feature_linear/train_final.py` on all 316 official
training cases. Verify the checkpoint contract and preprocessing round trip.
Do not load the official test until a separate locked-test evaluation entry
point has been reviewed.

## Supported active files

- `README.md`
- `docs/STATUS.md`
- `data/README.md`
- `data/processing/finger_movements/prepare_finger_movements.py`
- `models/finger_movements/feature_linear/`
- `experiments/active/README.md`
- `results/README.md`
- `history/README.md`
