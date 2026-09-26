# Indy/Loco decoder

The next experiment is [Phase 16 parameter scaling](experiment/phase16_parameter_scaling/README.md):
modest width-80 TCN/GRU scaling with same-fold Midsize weight transfer and a
faster validation stopping rule. Data/folds/loss/optimizer settings stay fixed;
scripts and no-training checks are ready.

The retained deployment state combines the final Phase-13 neural checkpoints, six Phase-14
best-fold CubeAI packages, the Phase-15 30-fold Large PC evaluation, and the
completed six-bank firmware/GUI deployment integration. No new numbered model
phase was introduced for the deployment plumbing.

Six benchmark sessions are packaged under [`models/`](models/), with all five
cross-validation checkpoints in both Midsize and Large. One filename per
session includes `_best-test-fold` for inspection, but the paper result uses all
five validation-selected folds.

## Final paper-facing result

| Tier | Cross-validation state | Test R² |
|---|---|---:|
| Midsize | 30/30 folds complete | **0.7411 ± 0.0656** |
| Large | 30/30 PC exact-KNN folds complete; six selected-fold BCIMEM banks packed and integrated | **0.7498 ± 0.0632** |

Large is the same TCN+GRU neural base plus fold-specific GRU-hidden[49]
external residual memory. The old Phase-12 memlibs were archived because they
do not match the new checkpoints or seven-minute preprocessing contract.
The Large value is the Phase-15 exact-PC memory-quality result. Its 30
evaluation `.memlib` files remain PC archives. Separately, the six highlighted
deployment folds have been packed into firmware-compatible `BCIMEM1` images,
validated against the CM7 IVF policy, and installed in the GUI repository.

## Start here

- [`STATUS.md`](STATUS.md) — active phase and completion boundary
- [`models/manifest.json`](models/manifest.json) — machine-readable package index
- [`models/FINAL_MODEL_STATUS.md`](models/FINAL_MODEL_STATUS.md) — final result,
  definitions, and caveats
- [`models/CUBEAI_NEXT_PHASE.md`](models/CUBEAI_NEXT_PHASE.md) — completed
  conversion/deployment handoff and remaining board gates
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
folds × two tiers. It validates the model repository package; the deployed C
runtime and six `BCIMEM1` copies are owned by the firmware and GUI repositories.
