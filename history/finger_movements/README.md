# FingerMovements Experiment Archive

This archive preserves the completed Phase 1 model-selection program. Nothing
under this directory is an active Python dependency.

## Contents

```text
EXPERIMENT_LOG.md   chronological protocol, results, and decisions
STATUS.md           state at the end of Phase 1c
experiments/        exact Phase 1b/1c experiment scripts
results/            metrics, OOF predictions, learning curves, and figures
```

The scripts retain the paths they used while active. They are preserved for
provenance, not as supported active entry points.

## Final decision

Feature + Linear at 50 epochs was selected. Its mean OOF balanced accuracy was
60.05% across seeds 42, 43, and 44, with 0.36 percentage-point seed standard
deviation and a 59.84% worst-seed result. At the same duration, Tiny EEGNet
reached 59.18% with 3.01-point seed standard deviation and a 56.03% worst-seed
result. Regularized CSP + LDA was not retained.

The supported implementation is now under
[`models/finger_movements/feature_linear/`](../../models/finger_movements/feature_linear/README.md).
