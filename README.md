# Embedded Neural Signal Models

This repository develops small neural-signal models that can later run at low
latency on firmware.

FingerMovements is the active task: classify left- versus right-hand movement
from a 28-channel, 500 ms EEG segment. Phase 1 model selection is complete.
Feature + Linear at 50 epochs is the sole active pipeline; no final checkpoint
has been trained yet.

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
models/finger_movements/feature_linear/      sole active model
experiments/active/                          empty until a new experiment starts
results/                                     active results only
docs/STATUS.md                               current truth and next gate
history/finger_movements/                    completed EEG experiments/results
history/indy/                                completed Indy project
```

## Next gate

Train the frozen Feature + Linear pipeline once on all 316 official training
cases using:

```bash
python models/finger_movements/feature_linear/train_final.py
```

This produces the final checkpoint without loading the 100-case official test
split. Locked-test evaluation is a separate later step.
