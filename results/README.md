# Results

Store small, reproducible summaries in `metrics/` and large generated outputs in
`large/`.

- `metrics/`: compact JSON/CSV summaries that may be committed.
- `large/`: checkpoints, arrays, plots, and intermediate outputs; ignored by Git.

Each result should record the experiment script, configuration, session split,
random seed, and Git commit. Raw and processed neural data belong under `data/`,
not here.
