# Phase 4 — Architecture Reduction

- `phase4a_architecture_sweep.py`: architecture-only search with data split, sampling, learning rate, weight decay, and dropout frozen.
- `phase4b_five_seed_architecture_confirmation.py`: five-seed comparison of 64/64 and 48/48 models across five held-out months.
- `train_48x48_checkpoint.py`: final seed-43, 20-epoch checkpoint build.

The 48/48 model passed the predefined non-inferiority limits and reduced parameters from 78,786 to 45,266. It became the preferred standalone firmware candidate. Detector artifacts were not rebuilt for it and remain tied to 64/64.

Outputs are under `../../results/indy/phase4a_architecture_sweep/` and `phase4b_five_seed_confirmation/`.
