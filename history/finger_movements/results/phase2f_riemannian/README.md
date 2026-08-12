# Phase 2f Low-Dimensional Riemannian Comparison

This directory contains the completed TRAIN-only Phase 2f comparison between
the frozen Phase 2c causal CSSD + hierarchical LDA baseline and a
low-dimensional Riemannian tangent-space candidate.

The candidate reached 85.13% mean OOF balanced accuracy versus 83.99% for the
baseline, and improved worst-seed BA from 83.25% to 84.17%. It was not promoted
because seed 43 did not improve and both seed and fold variability increased.
All float32 OOF labels exactly matched float64, and official TEST was refused.

Decision: preserve these files as experimental evidence, keep the Phase 2c
400 ms checkpoint unchanged, stop model-family exploration, and proceed to
firmware deployment validation.

Key files:

- `phase2f_metrics.json`: protocol, aggregate metrics, resource estimates, and
  frozen promotion decision;
- `phase2f_seed_results.csv`: seed-level scores and confusion matrices;
- `phase2f_fold_results.csv`: fold-level metrics and numerical checks;
- `phase2f_oof_predictions.csv`: complete OOF predictions;
- `phase2f_error_complementarity.csv`: corrected and newly introduced errors;
- `phase2f_summary.png`: visual comparison.
