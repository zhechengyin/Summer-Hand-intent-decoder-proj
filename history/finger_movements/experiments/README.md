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
| `phasea2_cssd_lda.py` | Reproduced the paper-style BP/ERD CSSD + BP-trend hierarchy and ran TRAIN-only generalization diagnostics |
| `phase2b_cssd_stabilization.py` | Tested isolated CSSD changes; its saved result predates the official-MATLAB correction and is invalid-source provenance |
| `phase2b_combination_ablation.py` | Crossed 36 CSSD configurations on corrected official MATLAB TRAIN and selected the promoted model |
| `evaluate_archived_terminal_logistic_phase1.py` | Re-evaluated the frozen terminal-feature Logistic pipeline on corrected official MATLAB TRAIN |
| `phase2c_horizon_diagnostic.py` | Initial Phase 2c diagnostic that measured causal accuracy while accumulating 50--500 ms from a known epoch start; superseded by Phase 2c's past-only rolling interpretation |
| `phase2c_streaming_causal_cssd_lda.py` | Verified the final past-only 500 ms / 50 ms streaming interpretation, post-A invariance, and exact chunked filtering |
| `phase2c_bin_window_sweep.py` | Swept causal 200/300/400/500 ms feature rings and verified 10/20/50/100 ms bin equivalence, selecting 400/50 |
| `phase2d_evaluate_frozen_test.py` | Applied only the exact frozen Phase 2c 400 ms checkpoint to corrected official TEST and verified batch/streaming equivalence |
| `phase2e_lightweight_regularization_comparison.py` | Compared regularized CSSD, shrinkage LDA, ToeplitzLDA, and conditional fusion; no candidate was promoted |
| `phase2f_low_dimensional_riemannian.py` | Compared the Phase 2c baseline with a low-dimensional Riemannian tangent-space candidate; mean BA improved but the candidate failed all-seed and variability gates |

These files are frozen provenance. Active code must not import them. The
scripts themselves are retained even where an associated result was later
invalidated; consult `../EXPERIMENT_LOG.md` and result creation dates before
comparing metrics.
