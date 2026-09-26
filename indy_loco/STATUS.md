# Indy/Loco final status

**Current experiment phase:** Phase 16 — neural parameter scaling; scripts and
no-training validation complete, training and acceptance pending.

See [`experiment/phase16_parameter_scaling/README.md`](experiment/phase16_parameter_scaling/README.md).
Architecture sizes are configurable; the Phase-13 final Midsize preprocessing,
folds, loss and optimizer recipe remain frozen. The revised default is width 80
with 4 TCN blocks / 1 GRU layer, same-fold Midsize weight transfer, and the
user-authorized fast stopping rule (minimum 4 epochs, patience 3, 0.5% relative
validation improvement). No new training was run by the implementation task.

**Retained model phase:** Phase 15 — Large external-memory 30-fold PC validation complete.

**Current deployment state:** six selected-fold CubeAI bundles and six
firmware-compatible `BCIMEM1` banks are integrated on firmware branch `AI` and
GUI branch `deliverable3`.

**Status:** six sessions × five folds × two tiers remain packaged and
validated. The six highlighted best-fold neural checkpoints were converted
once (shared by Midsize and Large) and all passed X-CUBE-AI host/generated-C
accuracy replay. Their CubeAI diagnostic mean is **0.7941**, versus FP32
**0.7944**, but the paper-facing Midsize result remains the complete 30-fold
test R² of **0.7411 ± 0.0656**. Phase 15 rebuilt fold-specific PC evaluation
memlibs for all 30 folds; Large reached **0.7498 ± 0.0632**, a paired mean gain
of **+0.0086 R²**. The six selected-fold banks were subsequently packed and
replayed with the CM7 IVF policy: 0.794441 ABSENT versus 0.796320 READY. That
six-fold result is a deployment-format diagnostic, not the paper estimate.

The authoritative entry points are:

- `models/manifest.json` — machine-readable package index and phase gate
- `models/FINAL_MODEL_STATUS.md` — final metric definition and conclusion
- `experiment/phase14_cubeai_conversion/FINAL_REPORT.md` — conversion table
- `experiment/phase15_large_memory_validation/TECHNICAL_REPORT.md` — Large
  30-fold PC result and validity boundary
- `models/CUBEAI_NEXT_PHASE.md` — completed handoff and remaining board checklist
- `models/package_tools.py validate` — checkpoint and CubeAI package audit

The primary paper number is the mean across all validation-selected folds, not
the mean of six best-test-fold checkpoints. The filename marker
`_best-test-fold` is descriptive only.

Superseded best-fold deployment packages and old PC memlibs were moved to
`history/model_package_archive/phase12_best_test_fold_pre_phase13_final_2026-08-27/`.
The 30 evaluation memlibs remain under `experiment/` and are not firmware
images. The six highlighted deployment banks are stored once in the GUI repo;
the firmware contains the loader, validator, query builder, IVF search, and
neural fallback. Remaining empirical gates are Large board parity/latency and
rechecking zero coalesced predictions after the latest replay-scheduling fix.
