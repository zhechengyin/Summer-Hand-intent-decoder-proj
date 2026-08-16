# Configs

There is no separate active configuration file. The causal FingerMovements
configuration is declared beside the self-contained model under
`models/cssd_lda/` and recorded in its checkpoint metadata.
The initial horizon diagnostic is archived as
`history/experiments/phase2c_horizon_diagnostic.py`.

The promoted Phase 2c model freezes a 400 ms past-only feature ring and a
50 ms update interval. After cold reset, 100 ms of causal filter pre-roll makes
the first validated output occur at 500 ms; steady-state outputs then update
every 50 ms. Its state policy is explicit in
`models/cssd_lda/model.py`. The completed bin/window sweep is archived under
`history/`.

Completed experiment grids and their constants are preserved with their
scripts under `history/experiments/`. Indy configurations are isolated in the
sibling project at `indy_loco/configs/`.
