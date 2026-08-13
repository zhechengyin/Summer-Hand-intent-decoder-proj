# Phase 2e Lightweight Comparison

This corrected TRAIN-only experiment compared the frozen Phase 2c baseline with regularized CSSD, shrinkage LDA, their combination, block-Toeplitz LDA, and conditional fusion.

| Candidate | Mean OOF BA | Decision |
|---|---:|---|
| Phase 2c baseline | 83.99% | retained |
| Regularized CSSD | 84.10% | not promoted |
| ToeplitzLDA | 84.50% | not promoted |
| Baseline + Toeplitz nested fusion | 84.09% | not promoted |

No alternative consistently improved all seeds, worst-seed performance, and variability. The active checkpoint did not change.
