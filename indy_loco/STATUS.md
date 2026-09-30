# Indy/Loco final status

**Latest completed phase:** [Phase19](experiment/phase19_midsize_cuda/README.md), 2026-09-30.
CUDA benchmark complete; LayerNorm/ReLU saved 9.28% full-epoch time, below the
10% adoption threshold. GRU/combined candidates failed numerical update checks.
Width sweep complete: 90/90 training jobs, followed by 120/120 test evaluations
including original Midsize. Test R2 mean +/- sample SD: original64 0.741138 +/-
0.065598; width96 0.739618 +/- 0.065505; width128 0.743070 +/- 0.066986;
width256 0.741075 +/- 0.062379. Validation-selected checkpoints remained frozen
before test. No automatic model promotion. All sweep checkpoints are retained.

**Earlier phase:** Phase 17 — all three candidates completed and
their 30-fold checkpoints/predictions were verified. Test R2 mean +/- sample SD:
minGRU 0.6988 +/- 0.0750; Mamba-2 0.6865 +/- 0.0861; Transformer
0.6818 +/- 0.0818. Historical warm-started Midsize remains 0.7411 +/- 0.0656.
The user authorized a minGRU/Mamba-2 hyperparameter sweep on 2026-09-26:
12 configurations per model, two-stage validation selection, then two extra
training seeds per finalist (360 fits total). Architectures, preprocessing and
folds remain fixed. See the
[sweep protocol](experiment/phase17_architecture_comparison/sweep/README.md)
and live Phase17 results/sweep_v1/progress.json. Larger-model training and board
deployment are deferred. The original 16 checks and 7 new sweep checks pass.

**Previous experiment:** Phase 16 — neural parameter scaling; scripts and
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
