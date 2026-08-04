# Archived FingerMovements Feature + Logistic

This retired model was active after Phase 1d. Phase 1f replaced it with the
terminal low-pass representation. It is preserved only for provenance and must
not be imported by active code.

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

## Archive state

`C=1` is frozen for the current feature representation. Phase 1e found that
per-fold nested selection underperformed it, and the fixed-C upper refinement
found no meaningful, consistent improvement over it. Therefore:

- no final all-training-data checkpoint was created for this representation;
- the official test split was not evaluated with it.

`model.py` includes framework-independent linear inference so the eventual
weights can be used without PyTorch or scikit-learn at runtime. Scikit-learn is
used only by `fit_logistic` during offline training.
