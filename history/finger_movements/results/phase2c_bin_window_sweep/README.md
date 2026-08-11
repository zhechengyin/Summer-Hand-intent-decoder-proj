# Phase 2c causal bin/window sweep

This TRAIN-only experiment compared past-context windows of 200, 300, 400,
and 500 ms across seeds 42/43/44 with five whole-case folds per seed. It also
verified 10, 20, 50, and 100 ms causal streaming chunks. Official TEST was
refused and not loaded.

| Window | Mean OOF BA | Seed SD | Worst seed |
|---:|---:|---:|---:|
| 200 ms | 79.62% | 1.23 pp | 78.46% |
| 300 ms | 79.43% | 0.26 pp | 79.11% |
| **400 ms** | **83.99%** | **0.54 pp** | **83.25%** |
| 500 ms | 82.93% | 1.03 pp | 81.67% |

The 400 ms window is the frozen winner by the predeclared mean-BA rule,
improving mean BA by 1.05 points, worst-seed BA by 1.57 points, and seed SD by
0.49 points relative to 500 ms. Its seed results were 83.25%, 84.20%, and
84.52%. The sweep causally re-references BP to the oldest sample inside every
candidate ring; therefore the 400 ms classifier does not consume the removed
100 ms through its feature baseline.

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
