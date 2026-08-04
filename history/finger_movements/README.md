# FingerMovements Experiment Archive

This archive preserves the completed Phase 1b–1d research program. Nothing
under this directory is an active Python dependency.

## Contents

```text
EXPERIMENT_LOG.md   chronological protocol, results, and decisions
STATUS.md           state at the end of Phase 1d
experiments/        exact Phase 1b/1c/1d experiment scripts
models/             retired AdamW + dropout Feature + Linear implementation
results/            metrics, OOF predictions, learning curves, and figures
```

The scripts retain the paths they used while active. They are preserved for
provenance, not as supported active entry points.

## Final decision

Phase 1c initially selected Feature + Linear trained with AdamW and dropout at
60.05% mean OOF balanced accuracy. Phase 1d found no explicit processed-data,
label-transcription, duplicate, or fold-overlap failure, then compared
classifiers on the same 196 features. L2 Logistic Regression with `C=1` reached
64.37%, with 1.50 percentage-point seed standard deviation and a 62.68%
worst-seed result. It replaced AdamW training as the active candidate.

The supported implementation is now under
[`models/finger_movements/feature_logistic/`](../../models/finger_movements/feature_logistic/README.md).
Its regularization value remains provisional until Phase 1e nested-CV.
