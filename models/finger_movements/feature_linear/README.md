# FingerMovements Feature + Linear

This is the only active model. Phase 1c selected it over Tiny EEGNet and
regularized CSP + LDA.

## Contract

- Input: one or more raw EEG cases with shape `(cases, 28, 50)` and `float32`
  values.
- Sampling: 100 Hz; each case spans 500 ms.
- Output: two logits ordered as `left=0`, `right=1`.
- Training data: all 316 official training cases after model selection is
  frozen.
- Official test: not loaded by the training entry point.

## Pipeline

Training-derived channel normalization is followed by seven deterministic
features per channel: mean, standard deviation, mean square, and power in
1–4 Hz, 4–8 Hz, 8–13 Hz, and 13–30 Hz. The resulting 196 values are standardized
using training-derived statistics and passed to a dropout-regularized linear
classifier.

The frozen training policy is seed 42, 50 epochs, AdamW, learning rate 0.001,
weight decay 0.0001, dropout 0.25, and batch size 32.

## Final training

No final checkpoint exists yet because the Phase 1 experiments trained only
cross-validation fold models. Generate the single all-training-data checkpoint
from the repository root with:

```bash
python models/finger_movements/feature_linear/train_final.py
```

The checkpoint is written to
`models/finger_movements/feature_linear/checkpoints/feature_linear_seed42_epoch50.pt`.
It contains the model weights and the training-derived preprocessing arrays
required to reproduce inference. It does not contain or use official test
data.
