# Phase 3 detector history

This folder preserves the completed detector-development runners and their
regression checks:

- `phase3a_drift_detector.py`: label-free raw-count and simplified-MINDFUL
  baseline;
- `phase3b_leave_one_month_out.py`: five strict pre-January held-month decoder
  folds;
- `phase3c_decoder_state_detector.py`: frozen-decoder hidden/output analysis
  and generation of the final development artifacts;
- `test_*.py`: protocol and numerical regression evidence used before archival.

These files explain how the retained Phase 3 results were produced. They are
not active runtime dependencies and should not be imported by model code.

The active path is now `models/indy_32ch/runtime.py`. It loads the frozen model
checkpoint and both fitted detector artifacts, makes one compatibility decision
after the first 60 seconds, and releases only post-warm-up decoder output when
the decision permits it.

The strict outer-fold result caught both known pre-January negative-R² sessions.
The final artifact uses the one retained active checkpoint, which had already
seen those sessions during model training. At artifact level it still abstains
`indy_20161013_03`, but `indy_20160630_01` passes. This distinction is why the
runtime remains a development candidate pending prospective data.
