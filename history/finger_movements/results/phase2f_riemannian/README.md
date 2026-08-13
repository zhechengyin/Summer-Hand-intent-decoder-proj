# Phase 2f Low-Dimensional Riemannian Comparison

This corrected TRAIN-only experiment compared the causal Phase 2c baseline with a low-dimensional tangent-space classifier on identical folds.

| Model | Mean OOF BA | Seed SD | Worst seed |
|---|---:|---:|---:|
| Phase 2c baseline | 83.99% | 0.54 pp | 83.25% |
| Riemannian candidate | 85.13% | 1.14 pp | 84.17% |

Seed changes were +1.25, -0.03, and +2.21 points for seeds 42–44. Despite higher mean and worst-seed BA, seed 43 did not improve and variability increased. The candidate failed the predefined promotion rule; no checkpoint was created.
