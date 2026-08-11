# FingerMovements causal CSSD + hierarchical LDA

This is the archived Phase 2c 500 ms causal baseline. It predicts from the
500 ms EEG interval ending at the current point A. The implementation is
strictly causal: temporal filters run left-to-right, samples after A are never
used, and the streaming API carries IIR state plus a rolling 500 ms buffer.

## Frozen baseline configuration

- 28 EEG channels sampled at 100 Hz;
- 500 ms past-context window (50 samples);
- 50 ms streaming update (5 new samples);
- fourth-order causal 0--7 Hz BP filter;
- fourth-order causal 10--33 Hz ERD filter;
- empirical CSSD covariance with per-trial trace normalization;
- one BP and one ERD spatial pattern per class;
- BP, ERD, and BP-trend branch LDAs followed by LDA fusion.

TRAIN-only repeated five-fold validation across seeds 42/43/44 produced
82.93% mean OOF balanced accuracy, 1.03 percentage-point seed standard
deviation, and 81.67% worst-seed balanced accuracy.

## Checkpoint

The active checkpoint is:

```text
checkpoints/finger_movements_cssd_lda_phase2c_causal.npz
```

Its adjacent `.metrics.json` records the exact checkpoint hash, source-data
hash, development evidence, full-TRAIN fit diagnostics, reload verification,
and streaming-equivalence verification.

Checkpoint SHA-256:

```text
d92c23f7e6f8722d568d1b31963eab1328d5367ba32764b676d1ae0d73aaefd4
```

The apparent all-TRAIN balanced accuracy is 88.30%. This number is only a fit
diagnostic; the frozen generalization estimate remains 82.93% mean OOF BA.

Rebuild it from official TRAIN only:

```bash
python history/finger_movements/models/cssd_lda_causal_500ms_phase2c/train_checkpoint.py
```

The entry point refuses any data path containing `test`. Apparent metrics on
the 316 fitting cases are diagnostics, not held-out performance.

## Deployment boundary

The model is causal, but the official dataset contains isolated 500 ms epochs
rather than continuous EEG. Real continuous recordings are still required to
validate long-running filter state, overlapping windows, and rest/no-intent
behavior before firmware deployment.

The previous zero-phase Phase 2b implementation and checkpoint are preserved
under `history/finger_movements/models/cssd_lda_offline_phase2b/`.
