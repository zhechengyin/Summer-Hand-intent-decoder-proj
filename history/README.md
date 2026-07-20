# Historical executable files

This folder holds superseded experiment scripts that are retained only so the
project history stays understandable. Nothing under `history/` is imported by
the active model or used by the current Optuna sweep.

## `indy_32ch/`

- `train_chronological_baseline.py` is the earlier 29/4/4 diagnostic baseline.
  It printed January test metrics during training, so it is not a valid model-
  selection entry point.
- `train_sampling_comparison.py` is the completed window/session/month study
  from LOG-097 and LOG-098. It established session-balanced sampling as the
  frozen policy.
- `sampling_comparison_test.py` preserves the old sampler tests. It is outside
  `tests/`, so it is not part of the active test suite.

Their numerical outcomes and original commands remain in
`docs/history/EXPERIMENT_LOG.md`, while the small result JSON files remain in
`results/metrics/` as decision evidence.

## Current entry point

Run the self-contained Phase-1 sweep from the repository root:

```bash
python models/indy_32ch/sweep_phase1_optuna.py
```

Do not import code from this folder into new scripts. If an old direction is
reopened, copy the necessary idea into a new active implementation and document
the new protocol explicitly.
