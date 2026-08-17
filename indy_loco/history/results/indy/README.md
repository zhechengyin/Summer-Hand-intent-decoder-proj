# Indy Results Index

Each directory contains the immutable outputs for one archived phase.

| Directory | Contents |
|---|---|
| `phase0a_data_audit/` | session and month data audit |
| `phase0b_sampler_selection/` | window/session/month sampling comparison |
| `phase1e_seed_crosscheck/` | final hyperparameter seed check |
| `phase2_locked_january/` | consumed January evaluation |
| `phase3a_drift_detector/` | label-free detector scores and references |
| `phase3b_leave_one_month_out/` | held-month decoder checkpoints and metrics |
| `phase3c_decoder_state_detector/` | decoder-state detector scores and gate metadata |
| `phase4a_architecture_sweep/` | architecture search database and summaries |
| `phase4b_five_seed_confirmation/` | 64/64 versus 48/48 confirmation |
| `phase5a_64channel_width_comparison/` | exploratory 64-channel comparison |
| `phase5_64channel_detector_filtered_sweep/` | confirmed 64-channel tuning and detector-filter ablation |

Use the JSON/CSV files as the source of truth; figures are summaries. See `../../EXPERIMENT_LOG.md` for interpretation and validity limits.
