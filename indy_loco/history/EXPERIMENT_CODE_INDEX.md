# Indy Experiment Code Index

All paths in the table are relative to this `history/` directory, except the
retained model checkpoint path, which is relative to the Indy project root.

| Phase | Script | Primary output |
|---|---|---|
| 0a | `experiments/active_at_archive/phase0a_data_audit.py` | `results/indy/phase0a_data_audit/` |
| 0b | archived sampler script is not retained separately | `results/indy/phase0b_sampler_selection/` |
| 1a | `experiments/phase1/phase1a_optuna.py` | historical tuning artifacts |
| 1b | `experiments/phase1/phase1b_regularization_grid.py` | historical tuning artifacts |
| 1c | `experiments/phase1/phase1c_wd_upper_grid.py` | historical tuning artifacts |
| 1d | `experiments/phase1/phase1d_seed_confirmation.py` | historical tuning artifacts |
| 1e | `experiments/phase1/phase1e_seed_crosscheck.py` | `results/indy/phase1e_seed_crosscheck/` |
| 2 | `experiments/phase2/phase2_locked_january.py` | `results/indy/phase2_locked_january/` |
| 3a | `experiments/phase3/phase3a_drift_detector.py` | `results/indy/phase3a_drift_detector/` |
| 3b | `experiments/phase3/phase3b_leave_one_month_out.py` | `results/indy/phase3b_leave_one_month_out/` |
| 3c | `experiments/phase3/phase3c_decoder_state_detector.py` | `results/indy/phase3c_decoder_state_detector/` |
| 4a | `experiments/phase4/phase4a_architecture_sweep.py` | `results/indy/phase4a_architecture_sweep/` |
| 4b | `experiments/phase4/phase4b_five_seed_architecture_confirmation.py` | `results/indy/phase4b_five_seed_confirmation/` |
| 4c | `experiments/phase4/train_48x48_checkpoint.py` | `../models/indy_32ch/48x48checkpoint.pt` |
| 5a | `experiments/active_at_archive/phase5a_64channel_width_comparison.py` | `results/indy/phase5a_64channel_width_comparison/` |
| 5 | `experiments/phase5/phase5_64channel_detector_filtered_sweep.py` | `results/indy/phase5_64channel_detector_filtered_sweep/` |

Scripts under `experiments/active_at_archive/` were active immediately before the whole Indy project was archived; the directory name does not mean they are active now.

Phase 5 was completed after that earlier project archive. Phase 6 is the current
active experiment and is intentionally not listed as historical code.
