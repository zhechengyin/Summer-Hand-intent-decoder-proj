# Models

The active model is:

```text
models/finger_movements/cssd_lda/
```

It is the Phase 2b winner on corrected official MATLAB TRAIN data: empirical
CSSD covariance, per-trial trace normalization, one BP and one ERD/F2 spatial
pattern per class, three branch LDAs, and final LDA fusion. Its mean OOF
balanced accuracy was 86.72% across seeds 42/43/44 with five folds per seed.

The directory contains a self-contained implementation, an all-TRAIN fitting
entry point, a verified NPZ checkpoint, and machine-readable checkpoint
metrics. Active code does not import from `history/`.

This is an offline research model. Its zero-phase temporal filters are
non-causal, so it is not yet the firmware model. Retired models and their old
checkpoints remain under `history/finger_movements/models/`.
