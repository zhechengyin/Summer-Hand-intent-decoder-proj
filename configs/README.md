# Configs

There is no separate active configuration file. The causal FingerMovements
configuration is declared beside the self-contained model under
`models/finger_movements/cssd_lda/` and recorded in its checkpoint metadata.
The initial horizon diagnostic is archived as
`history/finger_movements/experiments/phase2c_horizon_diagnostic.py`.

The promoted Phase 2c baseline freezes a 500 ms past-only ring buffer and a
50 ms update interval. Its streaming state policy and endpoint feature windows
are explicit in `models/finger_movements/cssd_lda/model.py`. The active sweep
tests bins 10/20/50/100 ms and windows 200/300/400/500 ms in
`experiments/active/phase2c_bin_window_sweep.py`.

Completed experiment grids and their constants are preserved with their
scripts under `history/finger_movements/experiments/`. Retired Indy
configurations remain under `history/indy/configs/`.
