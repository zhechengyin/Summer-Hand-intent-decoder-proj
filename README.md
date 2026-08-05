# Embedded Neural Signal Models

This repository develops small neural-signal models that can later run at low
latency on firmware.

The first FingerMovements research direction is complete and archived. It
classified left- versus right-hand movement from a 28-channel, 500 ms EEG
segment sampled at 100 Hz. Its frozen terminal low-pass + Logistic Regression
pipeline achieved 68.89% mean out-of-fold balanced accuracy and 62.10%
balanced accuracy on the one-time official 100-case test.

There is currently no active experiment or promoted model. The repository is
at a clean direction-change boundary. The FingerMovements raw and processed
data remain available for the next experiment; completed code, results, model,
and checkpoint are preserved under `history/finger_movements/`.

## Project rules

1. Every model has one explicit input contract and one explicit target.
2. Datasets with different labels are not merged merely because they share a
   signal modality.
3. Learned preprocessing is fitted from training folds only.
4. Active code must not import from `history/`.
5. The FingerMovements official test has been opened once. Future model
   selection must use only the 316-case training split; official-test results
   are post-hoc comparisons, not a new tuning signal.

## Repository layout

```text
data/
  raw/                                      immutable source dataset
  processed/finger_movements/               model-ready official splits
  processing/finger_movements/              supported conversion code
experiments/active/                          empty until a new experiment is registered
models/                                      empty of promoted candidates
results/                                     empty of active experiment results
docs/STATUS.md                               current truth and next decision gate
history/finger_movements/                    completed EEG direction and checkpoint
history/indy/                                completed Indy project
```

## Next gate

Define and register the next representation experiment before adding code. The
current candidate direction is to compare the archived terminal-feature
baseline with physiologically motivated low-frequency potential, ERD, and
training-fold-only spatial features inspired by CSSD/FDA. No new phase name or
configuration has been frozen yet.
