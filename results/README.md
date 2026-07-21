# Results

Store small, reproducible summaries in `metrics/` and large generated outputs in
`large/`.

- `metrics/`: compact JSON/CSV summaries that may be committed.
- `large/`: checkpoints, arrays, plots, and intermediate outputs; ignored by Git.

Each result should record the experiment script, configuration, session split,
random seed, and Git commit. Raw and processed neural data belong under `data/`,
not here.

## Current model-selection evidence

- `metrics/indy_32ch_sampling_seed_sweep.json`: complete CPU seed 42/43/44
  comparison of window-, session-, and month-balanced training, including all
  epoch histories, selected checkpoints, per-session metrics, cross-seed mean/SD,
  and per-seed wins.
- `figures/indy_32ch_sampling_seed_sweep.png`: mean training/validation loss and
  R² curves with +/-1 sample-SD bands across the three seeds.
- Decision: session-balanced sampling is frozen for subsequent hyperparameter
  sweeps; January test remains locked and was not loaded.
- `metrics/indy_32ch_phase1_optuna.json`: completed 40-trial Phase-1 histories,
  hyperparameters, pruning states, pooled/macro/worst-session validation metrics,
  and protocol metadata.
- `figures/indy_32ch_phase1_optuna.png`: completed-trial objective progression
  and learning-rate/weight-decay/dropout relationships.
- `large/indy_32ch_phase1_optuna.db`: complete Optuna study (29 complete,
  11 pruned, zero failed).
- `large/indy_32ch_phase1_best_checkpoint.pt`: trial-32 epoch-7 checkpoint;
  Phase-1 validation loss 0.482066 and pooled R² 0.557579. It is a confirmation
  candidate, not a promoted or test-evaluated model.

Retired EEG/fNIRS binaries and unversioned historical iter logs are not retained.
Their reported outcomes and caveats are indexed in
`docs/history/ARCHIVE_RETIREMENT.md`.
