# History

This directory contains only completed code needed to explain how the current
frozen model was selected:

- `phase1/sweep_phase1_optuna.py`
- `phase1/sweep_phase1b_regularization_grid.py`
- `phase1/sweep_phase1c_wd_upper_grid.py`
- `phase1/sweep_phase1d_seed_confirmation.py`
- `phase1/sweep_phase1e_seed_crosscheck.py`
- `phase1/results/phase1e_seed_crosscheck.json`
- `locked_test/evaluate_locked_january.py`

The Phase-1 scripts cover the completed Optuna search, boundary grids and
five-seed confirmation. The locked-test runner produced the one-shot January
result on 2026-07-22.

These files are provenance, not active entry points. Do not run or import them.
They may reference deleted intermediate databases or non-selected checkpoints.
The frozen configuration is in `configs/indy_32ch.yaml`; current conclusions are
in `docs/history/EXPERIMENT_LOG.md`.
