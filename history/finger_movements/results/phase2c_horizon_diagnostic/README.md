# Phase 2c initial causal horizon diagnostic

This directory contains the completed TRAIN-only evaluation of the 50 ms-bin
strict-causal CSSD + hierarchical LDA candidate.

Headline result at the final 500 ms horizon:

- mean OOF balanced accuracy: 82.93%;
- seed standard deviation: 1.03 percentage points;
- worst-seed balanced accuracy: 81.67%;
- delta from the frozen zero-phase reference: -3.79 percentage points;
- future-replacement invariance: exact zero error at every horizon;
- official TEST: refused and not loaded.

Files:

- `phase2c_horizon_metrics.json`: complete protocol, causality contract, data hash,
  seed results, and horizon summary;
- `phase2c_horizon_summary.csv`: accuracy/latency curve from 50 to 500 ms;
- `phase2c_horizon_seed_results.csv`: OOF metrics per seed and horizon;
- `phase2c_horizon_fold_results.csv`: all fold-level metrics;
- `phase2c_horizon_predictions.csv`: per-case OOF predictions at every horizon;
- `phase2c_horizon_future_invariance.csv`: explicit future-replacement checks;
- `phase2c_horizon_accuracy.png`: summary figure.

This experiment does not create or replace a checkpoint. The frozen offline
checkpoint remains unchanged under `models/finger_movements/cssd_lda/`.
