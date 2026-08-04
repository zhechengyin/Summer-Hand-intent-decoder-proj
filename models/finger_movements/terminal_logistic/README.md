# FingerMovements Terminal Low-pass + Logistic

This is the sole active model frozen after Phase 1f.

## Input and target

- Input: `float32` EEG with shape `(cases, 28, 50)`.
- Sampling: 100 Hz; each case spans 500 ms and ends 130 ms before keypress.
- Labels: `left=0`, `right=1`.
- Output: one signed linear score; a non-negative score predicts right.

## Frozen preprocessing

All learned normalization values must come from training data only.

1. Normalize each of the 28 channels with training-derived mean and standard
   deviation.
2. Apply a second-order causal 5 Hz low-pass IIR. Each fixed 500 ms trial is
   initialized from its first sample; firmware must reproduce this policy.
3. Extract 252 terminal features in this exact order:
   - five terminal samples per channel: 140 values;
   - final 5-, 10-, and 20-sample mean per channel: 84 values;
   - final 20-sample least-squares slope per channel: 28 values.
4. Standardize the 252 features with training-derived statistics.
5. Apply L2 Logistic Regression with `C=1` and one linear output.

The inference model has 252 learned weights and one bias. Offline training uses
scikit-learn; `TerminalLogistic` provides framework-independent inference.

## Evidence and freeze state

Phase 1f used seeds 42, 43, and 44 with identical stratified five-fold splits.
The frozen pipeline achieved 68.89% mean OOF balanced accuracy, 0.92
percentage-point seed standard deviation, and 68.35% worst-seed balanced
accuracy. The retired 196-feature Logistic baseline achieved 64.37%, 1.50
percentage points, and 62.68%, respectively.

The model architecture, feature contract, causal filter, classifier family,
and `C=1` are frozen. No final all-training-data checkpoint exists yet, and the
official 100-case test split remains locked.
