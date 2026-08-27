# Indy/Loco final status

**Current phase:** Phase 13 Round 3 — final Python checkpoint package, pre-CubeAI.

**Status:** six sessions × five folds × two tiers are packaged and validated.
Midsize has a paper-facing 30-fold test R² of **0.7411 ± 0.0656**. Large has the
same 30 neural checkpoints, but its 30 fold-specific GRU-memory banks and
corrected R² are still pending. CubeAI conversion was intentionally not run in
this phase.

The authoritative entry points are:

- `models/manifest.json` — machine-readable package index and phase gate
- `models/FINAL_MODEL_STATUS.md` — final metric definition and conclusion
- `models/CUBEAI_NEXT_PHASE.md` — next conversion and Large-memory checklist
- `models/package_tools.py validate` — 60-file integrity and protocol audit

The primary paper number is the mean across all validation-selected folds, not
the mean of six best-test-fold checkpoints. The filename marker
`_best-test-fold` is descriptive only.

Superseded best-fold deployment packages and old PC memlibs were moved to
`history/model_package_archive/phase12_best_test_fold_pre_phase13_final_2026-08-27/`.
Firmware, GUI, generated CubeAI files, and board deployment were not changed.
