# FingerMovements Experiment Archive

This archive preserves the complete first FingerMovements research direction,
Phase 1b through Phase 1h. Nothing under this directory is an active Python
dependency.

## Contents

```text
EXPERIMENT_LOG.md   chronological protocol, results, and decisions
STATUS.md           final state at the 2026-08-05 archive boundary
experiments/        exact Phase 1b–1h experiment and inference scripts
models/             retired implementations and frozen final checkpoint
results/            metrics, predictions, learning curves, and figures
```

The scripts retain the paths used while they were active. They are preserved
for provenance and are not supported active entry points.

## Final outcome

The selected pipeline was a causal 5 Hz low-pass, 252 terminal ABC features,
and L2 Logistic Regression with `C=1`. It achieved 68.89% mean OOF balanced
accuracy during model selection. The exact all-training-data checkpoint then
reached 62.00% accuracy and 62.10% balanced accuracy on the one-time official
100-case test.

This is retained as a reproducible baseline, not promoted as the final
firmware model. The 6.78 percentage-point OOF-to-test balanced-accuracy drop
and the published headroom on this dataset motivate a change in EEG feature
representation rather than further tuning of the same terminal features.
