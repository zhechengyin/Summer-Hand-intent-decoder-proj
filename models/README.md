# Models

The active model is:

```text
models/finger_movements/cssd_lda/
```

It is the Phase 2c strictly causal successor to the Phase 2b offline model:
empirical CSSD covariance, per-trial trace normalization, one BP and one ERD/F2
spatial pattern per class, three branch LDAs, and final LDA fusion. Its mean
OOF balanced accuracy was 82.93% across seeds 42/43/44 with five folds per
seed.

The directory contains a self-contained implementation, an all-TRAIN fitting
entry point, a verified NPZ checkpoint, and machine-readable checkpoint
metrics. Active code does not import from `history/`.

This model uses causal temporal filters and a persistent streaming interface.
It remains a firmware candidate because continuous EEG and rest/no-intent
behavior have not yet been validated. The former zero-phase Phase 2b model and
other retired checkpoints remain under `history/finger_movements/models/`.
