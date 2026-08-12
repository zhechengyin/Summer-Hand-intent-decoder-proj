# Phase 2e lightweight CSSD/LDA comparison

This completed TRAIN-only experiment used the exact Phase 2c seeds 42/43/44,
five folds per seed, 400 ms causal feature ring, and 50 ms update contract.
Official TEST was refused.

ToeplitzLDA produced the highest mean OOF balanced accuracy at 84.50%, versus
83.99% for the current baseline. It did not qualify for promotion: seed 44
dropped by 1.28 percentage points, worst-seed BA was slightly lower, and seed
and fold variability increased. The conditional nested fusion reached 84.09%
and did not stabilize the effect.

Decision: keep the frozen Phase 2c empirical CSSD + SVD-LDA checkpoint.

Files:

- `phase2e_metrics.json`: complete protocol, summaries, resources, and gate;
- `phase2e_summary.csv`: model-level comparison;
- `phase2e_seed_results.csv`: one OOF result per model and seed;
- `phase2e_fold_results.csv`: all paired outer-fold results;
- `phase2e_oof_predictions.csv`: per-case float64 and float32 predictions;
- `phase2e_error_complementarity.csv`: paired error overlap diagnostics;
- `phase2e_resource_estimates.csv`: firmware parameter/RAM estimates;
- `phase2e_summary.png`: mean, worst-seed, and fold-variability overview.
