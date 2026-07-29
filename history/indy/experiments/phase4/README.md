# Phase 4 history

This folder contains completed architecture experiments and the one-run
checkpoint builder. These files are provenance, not active entry points.

- `phase4a_architecture_sweep.py`: architecture-only Optuna study using five
  complete pre-January held-month folds.
- `phase4b_five_seed_architecture_confirmation.py`: controlled 64/64 versus
  48/48 comparison across seeds 42--46 and the same five held months.
- `train_48x48_checkpoint.py`: fixed seed-43, 20-epoch build that selected
  epoch 10 by minimum December validation loss and wrote the retained 48/48
  checkpoint.

Phase 4a and 4b did not save weights. The last script was run only after 48/48
passed every preregistered non-inferiority guardrail. It never loaded January
and did not modify the protected 64/64 checkpoint.

Retained artifacts:

- `models/indy_32ch/64x64checkpoint.pt`: integrated detector-compatible model;
- `models/indy_32ch/48x48checkpoint.pt`: standalone firmware candidate;
- `results/indy/phase4a_architecture_sweep/`: Phase-4a evidence;
- `results/indy/phase4b_five_seed_confirmation/`: Phase-4b and build evidence.
