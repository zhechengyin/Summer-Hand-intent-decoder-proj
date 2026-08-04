# Archived Experiment Code

| Script | Purpose |
|---|---|
| `phase1b_finger_movements_baselines.py` | Compared Feature + Linear, Tiny MLP, Tiny EEGNet, and Tiny Multi-scale CNN for 20 epochs |
| `phase1c_representation_comparison.py` | Compared Feature + Linear, Tiny EEGNet, and regularized CSP + LDA |
| `phase1c_eegnet_epoch_check.py` | Extended Tiny EEGNet to 60 epochs and registered 20/30/40/50/60 milestones |
| `phase1c_feature_linear_50epoch_check.py` | Extended Feature + Linear to 50 epochs and made the equal-duration final comparison |
| `phase1d_data_sanity_checks.py` | Audited TRAIN.ts/NPZ agreement, labels, duplicates, folds, shuffled labels, and small-subset fitting |
| `phase1d_classifier_comparison.py` | Compared AdamW Linear, L2 Logistic Regression, Ridge, and Linear SVM on identical features/folds |

These files are frozen provenance. Active code must not import them.
