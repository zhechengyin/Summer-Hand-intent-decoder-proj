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

Retired EEG/fNIRS binaries and unversioned historical iter logs are not retained.
Their reported outcomes and caveats are indexed in
`docs/history/ARCHIVE_RETIREMENT.md`.
