# Archived FingerMovements Experiment Code

All scripts are frozen provenance and must not be imported by active code.

## Invalid-source history

The following scripts originally ran on the retired UEA conversion. Their stored metrics are invalid unless a corrected MATLAB rerun is explicitly recorded:

- `phase1b_finger_movements_baselines.py`
- `phase1c_*`
- `phase1d_*`
- `phase1e_*`
- `phase1f_low_frequency_factorial.py`
- `phase1g_terminal_feature_ablation.py`
- `phase1h_*`
- the original run of `phasea2_cssd_lda.py`
- `phase2b_cssd_stabilization.py`

## Corrected-data evidence

| Script | Purpose |
|---|---|
| `phasea2_cssd_lda.py` | corrected paper-style CSSD + hierarchical LDA and diagnostics |
| `phase2b_combination_ablation.py` | corrected 36-configuration CSSD ablation |
| `evaluate_archived_terminal_logistic_phase1.py` | corrected terminal-logistic control |
| `phase2c_horizon_diagnostic.py` | causal accuracy versus available history |
| `phase2c_streaming_causal_cssd_lda.py` | past-only streaming and future-invariance checks |
| `phase2c_bin_window_sweep.py` | 200–500 ms window and 10–100 ms bin comparison |
| `phase2d_evaluate_frozen_test.py` | pure inference with the frozen 400 ms checkpoint |
| `phase2e_lightweight_regularization_comparison.py` | regularized CSSD/LDA and ToeplitzLDA comparison |
| `phase2f_low_dimensional_riemannian.py` | Riemannian tangent-space comparison |

Consult `../EXPERIMENT_LOG.md` before citing any result. Reusing a script does not make its old output valid for a new data source.
