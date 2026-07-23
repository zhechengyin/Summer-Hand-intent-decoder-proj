# History

This directory contains only completed code needed to explain how the current
frozen model was selected:

- `phase1/phase1a_optuna.py`
- `phase1/phase1b_regularization_grid.py`
- `phase1/phase1c_wd_upper_grid.py`
- `phase1/phase1d_seed_confirmation.py`
- `phase1/phase1e_seed_crosscheck.py`
- `phase2/phase2_locked_january.py`
- `tests/test_causality.py`
- `tests/test_session_balanced_sampling.py`

The Phase-1 scripts cover the completed Optuna search, boundary grids and
five-seed confirmation. Their retained evidence is under matching
`results/indy/phase*/` directories. The Phase-2 runner produced the one-shot
January result on 2026-07-22. The archived tests document the causal and
session-balanced invariants verified before this cleanup.

These files are provenance, not active entry points. Do not run or import them.
They may reference deleted intermediate databases or non-selected checkpoints.
The frozen configuration is in `configs/indy_32ch.yaml`; current conclusions are
in `docs/history/EXPERIMENT_LOG.md`.
