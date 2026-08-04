# FingerMovements Experiment Archive

This archive preserves the completed Phase 1b–1f research program. Nothing
under this directory is an active Python dependency.

## Contents

```text
EXPERIMENT_LOG.md   chronological protocol, results, and decisions
STATUS.md           state at the end of Phase 1f
experiments/        exact Phase 1b–1f experiment scripts
models/             retired AdamW and 196-feature Logistic implementations
results/            metrics, OOF predictions, learning curves, and figures
```

The scripts retain the paths they used while active. They are preserved for
provenance, not as supported active entry points.

## Final decision

Phase 1d selected the 196-feature Logistic baseline at 64.37% mean OOF balanced
accuracy. Phase 1e retained `C=1`. Phase 1f then found that causal terminal
low-frequency features were more informative than the 196 whole-window
statistics. Terminal Low-pass + Logistic reached 68.89%, with 0.92
percentage-point seed standard deviation and a 68.35% worst-seed result.

The supported implementation is now under
[`models/finger_movements/terminal_logistic/`](../../models/finger_movements/terminal_logistic/README.md).
