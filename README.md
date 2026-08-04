# Embedded Neural Signal Models

This repository develops small neural-signal models that can later run at low
latency on firmware.

FingerMovements is the active task: classify left- versus right-hand movement
from a 28-channel, 500 ms EEG segment. Phase 1f selected and froze the active
Terminal Low-pass + Logistic model. It uses 252 causal low-frequency terminal
features and Logistic Regression with `C=1`. No final checkpoint has been
trained yet.

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
models/finger_movements/terminal_logistic/   sole frozen active model
experiments/active/                          no active experiment
results/                                     reserved for future active results
docs/STATUS.md                               current truth and next gate
history/finger_movements/                    completed Phase 1b–1f research
history/indy/                                completed Indy project
```

## Next gate

Commit the frozen Phase 1f state. The next gated action is to fit one final
checkpoint on all 316 official training cases and then evaluate the official
100-case test exactly once. Neither action has been performed yet.
