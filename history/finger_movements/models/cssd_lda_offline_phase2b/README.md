# FingerMovements CSSD + hierarchical LDA

This is the archived Phase 2b zero-phase research model selected from the
corrected official MATLAB TRAIN data. It is retained only as the 86.72%
offline reference and is not an active dependency.

## Frozen configuration

- 28 EEG channels, 50 samples per case at 100 Hz;
- fourth-order zero-phase 0--7 Hz BP filter;
- fourth-order zero-phase 10--33 Hz ERD filter;
- empirical CSSD covariance with per-trial trace normalization;
- one BP and one ERD spatial pattern per class;
- BP, ERD, and BP-trend branch LDAs;
- final LDA over the three branch scores.

Phase 2b selected this configuration at 86.72% mean OOF balanced accuracy,
0.68 percentage-point seed standard deviation, and 86.09% worst-seed balanced
accuracy across seeds 42/43/44 and five folds per seed.

## Checkpoint

`checkpoints/finger_movements_cssd_lda_phase2b.npz` contains every learned
spatial filter, branch/fusion scaler, LDA coefficient, channel order, temporal
filter coefficient, and training metadata. The adjacent `.metrics.json` file
records its hash, data hash, selection evidence, apparent all-TRAIN fit
diagnostics, and reload verification.

Frozen checkpoint SHA-256:

```text
1e95b1ab5eaf7277cadd658578ef343f67923fc2b197aec8e1231735163bbfa2
```

Rebuild it from the 316 official TRAIN cases:

```bash
python history/finger_movements/models/cssd_lda_offline_phase2b/train_checkpoint.py
```

The training entry point refuses paths containing `test`. Apparent metrics on
the same 316 fitting cases are diagnostics, not generalization estimates.

## Deployment boundary

This is an offline research checkpoint, not yet a firmware checkpoint. Its
paper-style `sosfiltfilt` preprocessing uses future samples inside each
500 ms trial and is therefore non-causal. Do not claim streaming or real-time
deployment until the filters are replaced with a causal implementation and
the causal candidate is revalidated using TRAIN-only cross-validation.
