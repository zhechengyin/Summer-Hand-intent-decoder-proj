# Archived FingerMovements Feature + Linear (AdamW)

This is the retired Phase 1c implementation. It was superseded in Phase 1d by
L2 Logistic Regression on the same 196 features. Nothing in this directory is
an active dependency.

## Contract

- Input: one or more raw EEG cases with shape `(cases, 28, 50)` and `float32`
  values.
- Sampling: 100 Hz; each case spans 500 ms.
- Output: two logits ordered as `left=0`, `right=1`.
- Intended final training data: all 316 official training cases.
- Official test: not loaded by the training entry point.

## Pipeline

Training-derived channel normalization is followed by seven deterministic
features per channel: mean, standard deviation, mean square, and power in
1–4 Hz, 4–8 Hz, 8–13 Hz, and 13–30 Hz. The resulting 196 values are standardized
using training-derived statistics and passed to a dropout-regularized linear
classifier.

The retired training policy is seed 42, 50 epochs, AdamW, learning rate 0.001,
weight decay 0.0001, dropout 0.25, and batch size 32.

## Final training

No final checkpoint was produced. `train_final.py` is retained only to document
the former contract and must not be used as an active entry point.
