# Indy/Loco decoder

The active model state is **Phase 15 — final Phase-13 neural checkpoints,
six Phase-14 best-fold CubeAI packages, and complete 30-fold Large PC memory
validation**.

Six benchmark sessions are packaged under [`models/`](models/), with all five
cross-validation checkpoints in both Midsize and Large. One filename per
session includes `_best-test-fold` for inspection, but the paper result uses all
five validation-selected folds.

## Final paper-facing result

| Tier | Cross-validation state | Test R² |
|---|---|---:|
| Midsize | 30/30 folds complete | **0.7411 ± 0.0656** |
| Large | 30/30 PC exact-KNN memory folds complete; firmware banks pending | **0.7498 ± 0.0632** |

Large is the same TCN+GRU neural base plus fold-specific GRU-hidden[49]
external residual memory. The old Phase-12 memlibs were archived because they
do not match the new checkpoints or seven-minute preprocessing contract.
The Large value is the Phase-15 PC memory-quality result; its 30 evaluation
memlibs are not firmware-compatible BCIMEM binaries.

## Start here

- [`STATUS.md`](STATUS.md) — active phase and completion boundary
- [`models/manifest.json`](models/manifest.json) — machine-readable package index
- [`models/FINAL_MODEL_STATUS.md`](models/FINAL_MODEL_STATUS.md) — final result,
  definitions, and caveats
- [`models/CUBEAI_NEXT_PHASE.md`](models/CUBEAI_NEXT_PHASE.md) — next conversion
  and Large-memory workflow
- [`experiment/phase13_deployment_validation/`](experiment/phase13_deployment_validation/)
  — training scripts, fold metrics, and checkpoints
- [`experiment/phase15_large_memory_validation/`](experiment/phase15_large_memory_validation/)
  — 30-fold Large PC replay, fold metrics, and evaluation memlibs
- [`history/`](history/) — archived experiments and superseded packages; never an
  active model-selection source

## Integrity check

```bash
.venv-deploy/bin/python indy_loco/models/package_tools.py validate
```

This loads and verifies 60 packaged checkpoint copies: six sessions × five
folds × two tiers. No CubeAI, generated C, firmware, GUI, or board artifact was
changed in this phase.
