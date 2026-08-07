# Project Status

Updated: 2026-08-05

## Current state

The first FingerMovements experiment direction is complete and archived.
Phase A2 is now active and tests a paper-style CSSD + hierarchical LDA
representation. There is no promoted model under `models/` yet.

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

## Phase A2 protocol

Phase A2 follows the three physiological branches from Wang et al. (2004):

- BP: zero-phase 0--7 Hz, points 44--47, one CSSD pattern per class;
- ERD: zero-phase 10--33 Hz, points 19--50, three CSSD patterns per class and
  absolute pooling from 192 to 24 values;
- BP trend: beginning and ending means on the paper's retained 19 channels.

The raw branch dimensions are 8, 24, and 38. Each branch is mapped to one LDA
score, then a final LDA combines the resulting three values. All spatial
filters, scalers, and LDA parameters use the current training fold only. Seeds
42, 43, and 44 use stratified five-fold cross-validation. TEST is refused.

The paper specifies zero-phase filters but not their family/order or a CSSD
numerical regularizer. Phase A2 freezes fourth-order Butterworth filters and a
`1e-6` covariance ridge. The paper used a perceptron for final fusion; Phase A2
uses LDA as requested. These differences are recorded rather than hidden.

The first complete TRAIN-only execution produced:

- mean OOF accuracy: 59.81%;
- mean OOF balanced accuracy: 59.81%;
- seed balanced-accuracy SD: 1.87 percentage points;
- worst-seed balanced accuracy: 57.90%;
- difference from archived terminal-Logistic OOF mean: -9.07 percentage
  points.

This initial result does not support replacing the archived baseline.

## Phase A2 generalization diagnosis

The diagnostic run used TRAIN only and did not change the standard Phase A2
predictions. Its prediction table contains all 316 cases for every seed,
fusion protocol, and branch subset, with no duplicate or missing cases.

Weighted outer-fold train-to-validation AUC was:

- BP CSSD: 61.77% to 52.59%, a 9.18-point gap;
- ERD CSSD: 83.18% to 55.79%, a 27.39-point gap;
- non-CSSD BP trend: 79.54% to 65.83%, a 13.71-point gap;
- three-score final fusion: 88.69% to 63.94%, a 24.75-point gap.

The seven-branch ablation shows that adding the CSSD branches does not improve
generalization. With inner-OOF fusion, balanced accuracy was 50.17% for BP
CSSD alone, 53.44% for ERD CSSD alone, 62.25% for BP trend alone, and 61.08%
for all three branches. Cross-fitting therefore repairs part of the fusion
bias but does not repair the CSSD representations.

Cross-fold subspace comparison also found weak CSSD stability. Mean subspace
cosine similarity was 69.33% for BP-left, 48.01% for BP-right, 75.78% for
ERD-left, and 61.15% for ERD-right. Some fold pairs were almost orthogonal,
despite the outer training folds sharing most cases. Across the three seeds,
102 of 316 cases changed predicted class, while 75 were always classified
incorrectly.

The current diagnosis is therefore spatial-filter instability plus branch
overfitting, not missing fold coverage and not primarily a final-fusion
problem. BP covariance condition numbers around 1,119--1,389 make covariance
estimation a plausible contributor. The next justified experiment is a
TRAIN-only covariance stabilization check for CSSD (shrinkage or stronger
regularization), judged by both held-out balanced accuracy and cross-fold
subspace stability. Do not tune final fusion first, and do not use official
TEST for this decision.

## Archived-direction decision

The archived pipeline remains the reproducible linear baseline, but 62% test
accuracy is not strong enough to promote it as the final firmware solution.
Classifier and regularization comparisons are considered complete for this
representation. The next useful change is the EEG representation, not another
small sweep of Logistic `C` or training duration.

Phase A2 has completed the initial representation test and its generalization
diagnosis. Do not open TEST or use it to choose the next representation.

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
- `experiments/active/phasea2_cssd_lda.py`
- `models/README.md`
- `results/README.md`
- `history/finger_movements/`
- `history/indy/`
