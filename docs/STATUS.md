# Project Status

Updated: 2026-08-04

## Active task

FingerMovements binary left/right classification is active. Each case contains
28 EEG channels and 50 samples at 100 Hz. The processed data preserves the
official 316-case train and 100-case test split.

Phase 1b through Phase 1f are complete and archived. The official test split
has never been loaded by the experiment scripts.

## Active model candidate

Terminal Low-pass + Logistic is the sole frozen active model:

- second-order causal 5 Hz low-pass filtering;
- five terminal samples, three terminal-window means, and one terminal slope
  per channel, producing 252 inputs;
- training-derived channel and feature normalization;
- one binary linear decision score with 252 weights and one bias;
- L2 Logistic Regression with `liblinear`;
- frozen regularization: `C=1`.

Phase 1f measured 68.89% mean OOF balanced accuracy across seeds 42, 43, and
44. Seed standard deviation was 0.92 percentage points and worst-seed balanced
accuracy was 68.35%. The retired 196-feature Logistic baseline reached 64.37%,
1.50 percentage points, and 62.68%, respectively. Terminal Logistic improved
all three seeds. Only the seed-43 paired comparison reached p<0.05, so the
selection is an engineering decision rather than a strong independent-session
significance claim.

## Artifact state

- Final all-training-data checkpoint: not yet trained.
- Official test evaluation: locked and not run.
- Active experiment: none.
- Frozen implementation:
  `models/finger_movements/terminal_logistic/`.
- Phase 1b–1f scripts, results, and retired models:
  `history/finger_movements/`.
- Structural limitation: trial-level recording-session IDs are unavailable, so
  random-fold results do not establish new-session generalization.
- Completed Indy project: `history/indy/`.

## Next gate

Commit the frozen model state. After explicit authorization, fit one final
model on all 316 training cases, save its preprocessing parameters and linear
weights, then evaluate the locked official test once. Do not use test results
to revise the frozen model.

## Supported active files

- `README.md`
- `docs/STATUS.md`
- `data/README.md`
- `data/processing/finger_movements/prepare_finger_movements.py`
- `models/finger_movements/terminal_logistic/`
- `experiments/active/README.md`
- `results/README.md`
- `history/README.md`
