# Embedded Neural Signal Models

This repository develops small neural-signal models that can later run at low
latency on firmware.

FingerMovements is the active task: classify left- versus right-hand movement
from a 28-channel, 500 ms EEG segment. Phase 1d selected the current Feature +
Logistic candidate over the former AdamW + dropout linear training path. Its
`C=1` regularization value still requires Phase 1e nested-CV confirmation. No
final checkpoint has been trained yet.

Completed research is preserved under `history/` and is never imported by
active code.

## Project rules

1. Every model has one explicit input contract and one explicit target.
2. Datasets with different labels are not merged merely because they share a
   signal modality.
3. Train, validation, and test boundaries are fixed before tuning.
4. The official test remains locked until the model and training policy are
   frozen.
5. Active code must not import from `history/`.

## Active repository layout

```text
data/
  raw/                                      immutable source dataset
  processed/finger_movements/               model-ready official splits
  processing/finger_movements/              supported conversion code
models/finger_movements/feature_logistic/    sole active model candidate
experiments/active/                          empty until Phase 1e is implemented
results/                                     active results only; currently empty
docs/STATUS.md                               current truth and next gate
history/finger_movements/                    completed EEG experiments/results
history/indy/                                completed Indy project
```

## Next gate

Implement Phase 1e to select Logistic Regression `C` with training-only nested
cross-validation. Keep the official 100-case test split locked and do not train
the all-training-data checkpoint until `C` is frozen.
