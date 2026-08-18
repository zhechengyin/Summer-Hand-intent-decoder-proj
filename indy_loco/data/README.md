# Indy and Loco Data

This directory preserves the complete primate-reaching data path. Indy supplies
the active cross-session Phase 6 experiment; Loco is prepared separately for
the session-local Phase 7 paper benchmark.

- `raw/indy_loco/indy/`: 37 original Zenodo MAT sessions. Treat as immutable source data.
- `raw/indy_loco/loco/`: original Zenodo Loco MAT sessions; Phase 7 requires the three official benchmark sessions. Treat completed MAT files as immutable source data.
- `processing/indy_loco/indy/`: the preparation notebook and causal target helper.
- `processing/indy_loco/loco/`: the reproducible 4 ms NeuroBench-compatible converter.
- `processed/indy_loco/indy/`: model-ready NPZ sessions split into 29 train, 4 validation, and 4 test files.
- `processed/indy_loco/loco/`: per-session 4 ms NPZ artifacts with reach boundaries and the official ordered split; the three benchmark sessions are complete.

The split is chronological: April–October 2016 for training, December 2016 for validation, and January 2017 for test. The January split has already been evaluated and is not an untouched holdout.

Do not edit source MAT files. If preprocessing is rerun, write to a new location and record the exact transformation and split.
