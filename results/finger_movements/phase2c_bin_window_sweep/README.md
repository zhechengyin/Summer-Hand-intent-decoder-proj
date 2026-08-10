# Phase 2c causal bin/window sweep

This TRAIN-only experiment compared past-context windows of 200, 300, 400,
and 500 ms across seeds 42/43/44 with five whole-case folds per seed. It also
verified 10, 20, 50, and 100 ms causal streaming chunks. Official TEST was
refused and not loaded.

| Window | Mean OOF BA | Seed SD | Worst seed |
|---:|---:|---:|---:|
| 200 ms | 80.17% | 1.56 pp | 78.17% |
| 300 ms | 82.81% | 1.30 pp | 81.33% |
| **400 ms** | **83.45%** | **1.07 pp** | **81.98%** |
| 500 ms | 82.93% | 1.03 pp | 81.67% |

The 400 ms window is the provisional winner by the predeclared mean-BA rule,
improving mean BA by 0.52 points and worst-seed BA by 0.31 points relative to
500 ms. It did not improve every seed: seed 42 decreased, while seeds 43 and
44 improved. The 500 ms / 50 ms checkpoint therefore remains frozen until the
400 ms result receives a dedicated confirmation.

All four bin sizes produced exactly identical causal filtered signals for the
same endpoint (maximum error 0). Bin size is therefore a firmware latency and
update-cadence choice, not an accuracy hyperparameter in this model.

Files:

- `phase2c_bin_window_metrics.json`: protocol and complete summary;
- `phase2c_bin_window_summary.csv`: aggregate grid;
- `phase2c_bin_window_seed_results.csv`: seed-level metrics;
- `phase2c_bin_window_fold_results.csv`: fold-level metrics;
- `phase2c_bin_window_predictions.csv`: all OOF predictions;
- `phase2c_bin_equivalence_checks.csv`: chunked-filter checks;
- `phase2c_bin_window_sweep.png`: summary heatmap.
