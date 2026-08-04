# Project Status

Updated: 2026-08-03

## Active task

FingerMovements binary left/right classification is active. Each case contains
28 EEG channels and 50 samples at 100 Hz. The processed data preserves the
official 316-case train and 100-case test split.

Phase 1d is complete. The processed/source, duplicate, fold-integrity,
shuffled-label, and small-subset checks found no explicit pipeline failure. On
identical features and folds, L2 Logistic Regression outperformed the former
AdamW linear training path, Ridge Classifier, and Linear SVM.

## Active model candidate

Feature + Logistic is the sole active candidate:

- seven deterministic features per channel, producing 196 inputs;
- training-derived channel and feature normalization;
- one binary linear decision score with 196 weights and one bias;
- L2 Logistic Regression with `liblinear`;
- current candidate regularization `C=1`.

Phase 1d measured 64.37% mean OOF balanced accuracy across seeds 42, 43,
and 44. Seed standard deviation was 1.50 percentage points and worst-seed
balanced accuracy was 62.68%. The retired AdamW + dropout linear path reached
60.05%, with a 59.84% worst-seed result.

All three seeds favored Logistic Regression, although only one per-seed paired
comparison reached p<0.05. The result selects the classifier family as the
active engineering candidate; it does not yet freeze `C`. The official test
was never loaded.

## Artifact state

- Final all-training-data checkpoint: not yet trained.
- Official test evaluation: locked and not run.
- Active experiments: none; Phase 1e is not yet implemented.
- Phase 1d audit: passed with the structural limitation that trial-level
  recording-session IDs are unavailable.
- Completed FingerMovements experiments and results:
  `history/finger_movements/`.
- Completed Indy project: `history/indy/`.

## Next gate

Implement Phase 1e nested cross-validation over Logistic Regression `C`, using
only each outer fold's training cases for inner selection. Freeze `C` only
after reviewing mean balanced accuracy, outer-fold stability, worst-seed
behavior, and the remaining recording-session limitation. Do not train the
final checkpoint or load the official test before that decision.

## Supported active files

- `README.md`
- `docs/STATUS.md`
- `data/README.md`
- `data/processing/finger_movements/prepare_finger_movements.py`
- `models/finger_movements/feature_logistic/`
- `experiments/active/README.md`
- `results/README.md`
- `history/README.md`
