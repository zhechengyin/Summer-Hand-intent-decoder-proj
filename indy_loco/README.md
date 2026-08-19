# Indy Loco Project

The historical decoder and detector work is preserved under `history/`. Phase 6
is complete and promotes a regularized 96-channel decoder as the strongest
validation candidate. Phase 7 completed all 30 six-session five-fold fits with
test R² `0.7056 ± 0.0722`; its session-local checkpoints are benchmark evidence
and do not replace the Phase 6 firmware candidate. Earlier checkpoints remain
available for comparison. Phase 8 found Indy test R² `0.7576 ± 0.0396` at
48 ms lookahead and `0.7554 ± 0.0397` at 100 ms. The same controlled
accuracy/latency comparison is now ready to run on the three Loco benchmark
sessions. Phase 9 then replayed two strictly causal cold-start policies for the
promoted Phase 6 checkpoint. On December validation, a continuous rolling
past-window reached pooled R² `0.7526`, versus `0.7021` for the original
50-bin block-reset protocol, and was frozen before one January test inference.

## Problem definition

The project decoded x/y fingertip velocity every 40 ms from intracortical spike
counts. Every input window contains the previous 50 bins (2 seconds). The
promoted Phase 6 model uses all 96 raw-count channels plus 96 causal EWMAs;
historical compact models use 32 selected raw-count channels plus 32 EWMAs.
Targets are two-dimensional velocity.

All preprocessing used past information only:

- channel selection and normalization were estimated from training data;
- each session used its first 60 seconds for causal calibration;
- EWMA features were updated forward in time;
- validation and test data never updated model weights.

## Retained systems

The strongest validation candidate is the 96-channel 64/64 TCN+GRU checkpoint
in `models/indy_96ch/phase6_96ch_64x64_checkpoint.pt`. It has 86,978 parameters
and achieved pooled December validation R² `0.7004 ± 0.0019` over seeds
42–44. Training used 0.20 paired channel dropout; the promoted seed-43 epoch-15
checkpoint reached pooled R² 0.7022 and macro R² 0.7041.

The 32-channel 48/48 checkpoint in `models/indy_32ch/48x48checkpoint.pt`
remains the smaller firmware reference. It reduced parameters from 78,786 to
45,266 while remaining non-inferior in its five-seed, leave-one-month-out
comparison.

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
- Phase 6 confirmed the 96-channel paired-dropout winner over seeds 42–44. December selected the model; January remained unloaded.
- Phase 9 selected the rolling calibration-seeded window using December only.
  The selected policy then reached pooled January R² `0.7277`; January did not
  participate in policy selection.

Start with `docs/STATUS.md` for the current state and
`history/EXPERIMENT_LOG.md` for the decision trail. Completed runners and
results are under `history/`; the retained implementations and checkpoints are
under `models/`; the completed Phase 6 reproduction runners remain under
`experiments/active/`.
