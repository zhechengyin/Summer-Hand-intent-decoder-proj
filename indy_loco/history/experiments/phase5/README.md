# Phase 5 archive

`phase5_64channel_detector_filtered_sweep.py` completed the 64-channel
hyperparameter sweep and paired detector-filter ablation.

The confirmed full-session baseline used all 29 training sessions with
LR 0.0009, weight decay 0.025, and dropout 0.10. Across seeds 42–44 it reached
pooled December validation R² `0.6575 ± 0.0080` and macro R²
`0.6627 ± 0.0076`.

Removing the two retrospective detector failures produced pooled R²
`0.6552 ± 0.0034`. Its mean result was not better, although its worst-session
floor improved. The project therefore retains all sessions for model training
and treats the detector as a runtime gate.

The script and all checkpoints, tables, metrics, and figures are historical
evidence. They are not imported by Phase 6.
