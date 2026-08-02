# Active Experiments

## Phase 1b — FingerMovements baseline comparison

`phase1b_finger_movements_baselines.py` compares four deployment-oriented
baselines:

1. deterministic EEG statistics and band powers with a linear classifier;
2. a temporally pooled Tiny MLP;
3. a compact EEGNet-style CNN;
4. a compact multi-scale temporal CNN.

Only the 316 official training cases are used. Seeds 42, 43, and 44 each run a
stratified 5-fold cross-validation, using identical folds for all models. The
100 official test cases are locked and are not loaded.

Fixed settings:

```text
epochs=20
optimizer=AdamW
learning_rate=0.001
weight_decay=0.0001
dropout=0.25
batch_size=32
checkpoint=none (evaluate the fixed final epoch)
augmentation=none
```

Every fold fits channel normalization from its training subset only. The
feature-linear baseline also fits feature standardization from that same
training subset.

Run from the repository root:

```bash
python experiments/active/phase1b_finger_movements_baselines.py
```

CPU is the deterministic default. To validate data loading and all four forward
passes without training:

```bash
python experiments/active/phase1b_finger_movements_baselines.py --validate-only
```

Outputs are written to
`results/finger_movements/phase1b_baseline_comparison/`: epoch, fold, and seed
CSV files; a JSON report; and an accuracy/model-size figure. Cross-validation
checkpoints are not retained.

Repeated cross-validation measures sensitivity to fold assignment and model
initialization. Because this dataset contains one subject and does not expose
session IDs, it cannot establish cross-subject or cross-day generalization.
