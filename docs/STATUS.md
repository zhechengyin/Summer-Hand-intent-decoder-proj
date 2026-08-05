# Project Status

Updated: 2026-08-05

## Current state

The first FingerMovements experiment direction is complete and archived. There
is no active training script and no promoted model under `models/`.

The reusable data contract remains unchanged:

- task: binary left/right movement classification;
- input: 28 EEG channels and 50 samples at 100 Hz per case;
- development data: 316 official training cases;
- official test: 100 cases, opened once during Phase 1h;
- processed split: `data/processed/finger_movements/`.

## Archived baseline

The completed direction used a second-order causal 5 Hz low-pass, 252 terminal
ABC features, training-derived normalization, and L2 Logistic Regression with
`C=1`.

- model-selection estimate: 68.89% mean OOF balanced accuracy;
- seed standard deviation: 0.92 percentage points;
- worst-seed OOF balanced accuracy: 68.35%;
- final checkpoint fit: all 316 official training cases;
- apparent training balanced accuracy: 78.49%, not a generalization estimate;
- official-test accuracy: 62.00%;
- official-test balanced accuracy: 62.10%;
- official-test macro-F1: 61.94%;
- official-test confusion matrix: `[[33, 16], [22, 29]]`;
- test minus development OOF balanced accuracy: -6.78 percentage points.

The checkpoint is retained unchanged at:

```text
history/finger_movements/models/terminal_logistic/checkpoints/finger_movements_terminal_logistic_phase1h.npz
```

SHA-256:

```text
f8fca725c3b638219bbd734257cd958779e595add2fe1118e1e78689bc120047
```

## Decision

The archived pipeline remains the reproducible linear baseline, but 62% test
accuracy is not strong enough to promote it as the final firmware solution.
Classifier and regularization comparisons are considered complete for this
representation. The next useful change is the EEG representation, not another
small sweep of Logistic `C` or training duration.

The proposed direction is to evaluate physiologically motivated low-frequency
potential, 10–33 Hz ERD, and training-fold-only spatial projections inspired
by CSSD/FDA. This direction is not yet an active experiment and has no assigned
phase label.

## Evaluation constraint after opening TEST

The official test result is final evidence for the archived checkpoint. It
must not be used to choose new filters, windows, spatial components,
classifiers, or thresholds. New candidates must be selected using only
training-only cross-validation. Because TEST has already been inspected,
future scores on it are post-hoc comparisons rather than a pristine locked
test. A new external holdout would be needed for a fully independent final
claim.

## Supported current files

- `README.md`
- `docs/STATUS.md`
- `data/README.md`
- `data/processing/finger_movements/prepare_finger_movements.py`
- `experiments/active/README.md`
- `models/README.md`
- `results/README.md`
- `history/finger_movements/`
- `history/indy/`
