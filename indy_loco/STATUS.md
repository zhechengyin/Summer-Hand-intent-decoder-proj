# Indy/Loco final status

**Current phase:** Phase 14 — six per-session best-fold CubeAI packages validated.

**Status:** six sessions × five folds × two tiers remain packaged and
validated. The six highlighted best-fold neural checkpoints were converted
once (shared by Midsize and Large) and all passed X-CUBE-AI host/generated-C
accuracy replay. Their CubeAI diagnostic mean is **0.7941**, versus FP32
**0.7944**, but the paper-facing Midsize result remains the complete 30-fold
test R² of **0.7411 ± 0.0656**. Large memlibs and corrected R² are pending.

The authoritative entry points are:

- `models/manifest.json` — machine-readable package index and phase gate
- `models/FINAL_MODEL_STATUS.md` — final metric definition and conclusion
- `experiment/phase14_cubeai_conversion/FINAL_REPORT.md` — conversion table
- `models/CUBEAI_NEXT_PHASE.md` — remaining firmware and Large-memory checklist
- `models/package_tools.py validate` — checkpoint and CubeAI package audit

The primary paper number is the mean across all validation-selected folds, not
the mean of six best-test-fold checkpoints. The filename marker
`_best-test-fold` is descriptive only.

Superseded best-fold deployment packages and old PC memlibs were moved to
`history/model_package_archive/phase12_best_test_fold_pre_phase13_final_2026-08-27/`.
Firmware, GUI, and board deployment were not changed. Generated CubeAI files
are present only in the active model package; the next step is Large memlib
construction and then explicit firmware/GUI integration.
