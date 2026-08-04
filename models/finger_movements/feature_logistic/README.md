# FingerMovements Feature + Logistic

This is the sole active model candidate after Phase 1d. It retains the selected
196-dimensional handcrafted EEG representation and replaces stochastic AdamW
+ dropout training with deterministic L2 Logistic Regression.

## Contract

- Input: `float32` EEG with shape `(cases, 28, 50)`.
- Sampling: 100 Hz; each case spans 500 ms.
- Labels: `left=0`, `right=1`.
- Output: one signed decision score; score below zero predicts left and score at
  or above zero predicts right.
- Trainable inference parameters: 196 weights and one bias.
- Official test: locked and not used during Phase 1d.

## Pipeline

Training-derived channel normalization is followed by seven deterministic
features per channel: mean, standard deviation, mean square, and power in
1–4 Hz, 4–8 Hz, 8–13 Hz, and 13–30 Hz. The 196 features are standardized using
training-derived statistics and passed to a binary linear decision function.

Phase 1d evaluated `C=1` with the `liblinear` solver. Across seeds 42, 43, and
44 with stratified five-fold cross-validation, mean OOF balanced accuracy was
64.37%, seed standard deviation was 1.50 percentage points, and worst-seed
balanced accuracy was 62.68%.

## Freeze state

`C=1` is the current best candidate, not the final frozen regularization value.
Phase 1e must select `C` using training-only nested cross-validation. Therefore:

- no final all-training-data checkpoint exists;
- there is intentionally no active final-training script yet;
- the official test split must remain locked;
- the archived AdamW implementation must not be restored as an active
  dependency.

`model.py` includes framework-independent linear inference so the eventual
weights can be used without PyTorch or scikit-learn at runtime. Scikit-learn is
used only by `fit_logistic` during offline training.
