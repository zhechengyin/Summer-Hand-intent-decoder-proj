# Indy Loco Project

The historical decoder and detector work is preserved under `history/`. Phase 5
is complete and archived; Phase 6 is the only active experiment and tests a
96-channel extension without changing retained checkpoints.

## Problem definition

The project decoded x/y fingertip velocity every 40 ms from intracortical spike counts. Each input window contained the previous 50 bins (2 seconds): 32 selected raw-count channels plus 32 causal EWMA features. Targets were two-dimensional velocity.

All preprocessing used past information only:

- channel selection and normalization were estimated from training data;
- each session used its first 60 seconds for causal calibration;
- EWMA features were updated forward in time;
- validation and test data never updated model weights.

## Retained system

The preferred standalone firmware candidate was the 48/48 TCN+GRU checkpoint in `models/indy_32ch/48x48checkpoint.pt`. It reduced parameters from 78,786 to 45,266 while remaining non-inferior in the five-seed, leave-one-month-out comparison.

The drift detector remained coupled to the older 64/64 representation. Its two layers were:

1. label-free 60-second firing-rate and channel-pattern checks;
2. frozen-decoder hidden-state and output-distribution checks.

The detector was retrospective safety research, not a validated production gate. January was inspected before the detector design was finalized, and no truly prospective sessions were collected.

## Validity boundaries

- The 29/4/4 split is chronological by session.
- December validation influenced model and checkpoint choices.
- January test was opened once in Phase 2 and cannot be called untouched afterward.
- Phase 3 leave-one-month-out results are the strongest cross-month robustness evidence.
- Phase 5 confirmed the 64-channel hyperparameters over seeds 42–44 and found no mean benefit from retrospective detector filtering.
- Phase 6 uses all 96 physical channels, seed 43, and the Phase 5 winner for one 20-epoch controlled run. January remains unloaded.

Start with `docs/STATUS.md` for the current state and
`history/EXPERIMENT_LOG.md` for the decision trail. Completed runners and
results are under `history/`; the retained implementation and checkpoints are
under `models/`; the current runner is under `experiments/active/`.
