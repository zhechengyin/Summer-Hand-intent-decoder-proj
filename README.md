# Embedded Neural Signal Models

This repository develops small neural-signal models that can later run at low
latency on firmware.

The first FingerMovements research direction is complete and archived. It
classified left- versus right-hand movement from a 28-channel, 500 ms EEG
segment sampled at 100 Hz. Its frozen terminal low-pass + Logistic Regression
pipeline achieved 68.89% mean out-of-fold balanced accuracy and 62.10%
balanced accuracy on the one-time official 100-case test.

Phase A2 is the active experiment. It evaluates a paper-style CSSD +
hierarchical LDA representation using low-frequency BP, 10--33 Hz ERD, and BP
trend features. The archived terminal Logistic model remains the comparison
baseline; there is not yet a promoted model for the new direction.

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
experiments/active/phasea2_cssd_lda.py       active TRAIN-only experiment
models/                                      empty of promoted candidates
results/finger_movements/phasea2_cssd_lda/   active Phase A2 results
docs/STATUS.md                               current truth and next decision gate
history/finger_movements/                    completed EEG direction and checkpoint
history/indy/                                completed Indy project
```

## Next gate

Phase A2's generalization diagnosis identifies unstable, overfitted CSSD
spatial filters as the main limitation. The next useful check is TRAIN-only
covariance stabilization for CSSD, evaluated by held-out balanced accuracy and
cross-fold subspace stability. The official test must remain untouched.
