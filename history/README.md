# History

This directory contains only completed code needed to explain how the current
frozen model was selected:

- `phase1/phase1a_optuna.py`
- `phase1/phase1b_regularization_grid.py`
- `phase1/phase1c_wd_upper_grid.py`
- `phase1/phase1d_seed_confirmation.py`
- `phase1/phase1e_seed_crosscheck.py`
- `phase2/phase2_locked_january.py`
- `phase3/phase3a_drift_detector.py`
- `phase3/phase3b_leave_one_month_out.py`
- `phase3/phase3c_decoder_state_detector.py`
- `phase3/test_*.py`
- `phase4/phase4a_architecture_sweep.py`
- `phase4/phase4b_five_seed_architecture_confirmation.py`
- `phase4/train_48x48_checkpoint.py`
- `tests/test_causality.py`
- `tests/test_session_balanced_sampling.py`

The Phase-1 scripts cover the completed Optuna search, boundary grids and
five-seed confirmation. Their retained evidence is under matching
`results/indy/phase*/` directories. The Phase-2 runner produced the one-shot
January result on 2026-07-22. Phase 3 contains the completed raw-count,
leave-one-month-out, and decoder-state detector studies. The archived tests
document the protocol invariants verified before each cleanup. Phase 4 contains
the completed architecture Optuna runner, five-seed non-inferiority
confirmation, and fixed 48/48 checkpoint builder. The Phase-4a and Phase-4b
runners saved no weights. The final builder created
`models/indy_32ch/48x48checkpoint.pt` after confirmation, while preserving
`models/indy_32ch/64x64checkpoint.pt`.

These files are provenance, not active entry points. Do not run or import them.
They may reference deleted intermediate databases or non-selected checkpoints.
The frozen configuration is in `configs/indy_32ch.yaml`; current conclusions are
in `docs/history/EXPERIMENT_LOG.md`.
