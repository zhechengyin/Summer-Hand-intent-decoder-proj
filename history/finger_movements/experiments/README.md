# Archived Experiment Code

| Script | Purpose |
|---|---|
| `phase1b_finger_movements_baselines.py` | Compared Feature + Linear, Tiny MLP, Tiny EEGNet, and Tiny Multi-scale CNN for 20 epochs |
| `phase1c_representation_comparison.py` | Compared Feature + Linear, Tiny EEGNet, and regularized CSP + LDA |
| `phase1c_eegnet_epoch_check.py` | Extended Tiny EEGNet to 60 epochs and registered 20/30/40/50/60 milestones |
| `phase1c_feature_linear_50epoch_check.py` | Extended Feature + Linear to 50 epochs and made the equal-duration final comparison |
| `phase1d_data_sanity_checks.py` | Audited TRAIN.ts/NPZ agreement, labels, duplicates, folds, shuffled labels, and small-subset fitting |
| `phase1d_classifier_comparison.py` | Compared AdamW Linear, L2 Logistic Regression, Ridge, and Linear SVM on identical features/folds |
| `phase1e_logistic_regularization_sweep.py` | Used nested CV to test broad Logistic regularization selection on the 196-feature representation |
| `phase1e_logistic_upper_regularization_sweep.py` | Refined fixed `C=1` through `C=5` and retained `C=1` |
| `phase1f_low_frequency_factorial.py` | Crossed four EEG representations with Logistic and shrinkage Fisher, selecting terminal low-pass + Logistic |
| `phase1g_terminal_feature_ablation.py` | Measured the standalone and joint contribution of terminal feature groups A, B, and C |
| `phase1h_train_final_checkpoint.py` | Fitted and verified the frozen pipeline once on all 316 official training cases |
| `phase1h_evaluate_locked_test.py` | Performed the authorized one-time pure inference on the 100-case official test |

These files are frozen provenance. Active code must not import them.
