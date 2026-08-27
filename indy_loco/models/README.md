# Final pre-CubeAI model packages

This directory is the authoritative Python checkpoint surface for the final
Phase-13 Round-3 handoff. It contains six sessions, five cross-validation folds
per session, and two system tiers.

```text
models/
├── midsize/<session>/{fold-1.pt,...,fold-5.pt,manifest.json}
├── large/<session>/{fold-1.pt,...,fold-5.pt,manifest.json}
├── midsize/model.py
├── midsize/runtime.py
├── manifest.json
└── package_tools.py
```

Exactly one checkpoint filename per session includes `_best-test-fold`. That
marker is for inspection and future deployment selection only. It must not be
used as the paper estimate. The paper-facing result is the mean and standard
deviation across all five validation-selected test folds.

## Tier definitions

- **Midsize:** the Phase-13 Round-3 TCN+GRU retrained for seven-minute
  calibration, continuous causal EWMA, rolling 50-bin windows, and output at
  timestep 49.
- **Large:** the identical neural checkpoint plus a future fold-specific
  GRU-hidden[49] residual-memory bank. The neural checkpoints are ready, but
  compatible memory libraries have not yet been rebuilt.

The archived Phase-12 `.memlib` files are deliberately absent from the active
Large folders. They were built against older checkpoints and a different
preprocessing/evaluation protocol and are not valid for these weights.

## Paper result

The final Midsize result is **R² 0.7411 ± 0.0656** over 30 folds (six sessions ×
five folds). This is validation-selected cross-validation evidence, not a mean
of six test-selected best folds. Large has no reportable five-fold corrected R²
until all 30 fold-specific memory banks are rebuilt and evaluated.

See [`FINAL_MODEL_STATUS.md`](FINAL_MODEL_STATUS.md) for the complete table and
validity limits.

## Validation

```bash
.venv-deploy/bin/python indy_loco/models/package_tools.py validate
```

The validator loads all 60 packaged checkpoint copies, checks identity,
selection policy, seven-minute preprocessing metadata, SHA-256 hashes,
Midsize/Large parity, and confirms that CubeAI conversion and Large memory
promotion have not been claimed prematurely.

## CubeAI boundary

No ONNX, H5, generated C, weights binary, or `.aibundle` was created or changed
in this phase. The next conversion phase must use each fold file plus the shared
[`midsize/model.py`](midsize/model.py). See
[`CUBEAI_NEXT_PHASE.md`](CUBEAI_NEXT_PHASE.md).

The superseded best-test-fold packages and their old PC memlibs are preserved
under
`../history/model_package_archive/phase12_best_test_fold_pre_phase13_final_2026-08-27/`.
