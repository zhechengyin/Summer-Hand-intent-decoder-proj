# Phase 14 — six best-fold CubeAI conversions

Phase 14 first gated the conversion on `indy_20160622_01` fold 5, then
converted the highlighted best-test fold for each of the six sessions. The
neural checkpoint and generated CubeAI package are shared by Midsize and
Large; they are stored once under `models/midsize/<session>/cubeai/fold-<n>/`.
Each Large session contains a reference manifest rather than duplicate weights.

All six packages passed:

- exact Phase-13 seven-minute calibration and rolling-window reconstruction;
- Python/Keras graph parity;
- X-CUBE-AI 10.2 host validation and generated-C encoder replay;
- held-out accuracy gate of no more than 0.01 mean-R² loss.

CubeAI does not support Keras GRU `return_state`, and its importer crashes on
`Cropping1D`. The compatible graph therefore exposes the complete GRU state
sequence `[1, 50, 64]`; hidden state 49 is exactly row 49. This keeps the
future Large-memory query available without a second GRU execution.

The six selected-fold diagnostic mean is `R² 0.7941` for the CubeAI-generated
C encoder plus FP32 GRU, versus `0.7944` for PyTorch FP32. It is deliberately
not the paper estimate because the folds were selected using test R².

The paper-facing Midsize result remains the full 30-fold value:

`R² 0.7411 ± 0.0656`.

Authoritative results:

- `results/best_fold_conversion_summary.json`
- `results/best_fold_conversion_summary.csv`
- `FINAL_REPORT.md`

Re-run one selected fold from the repository root with:

```bash
.venv-deploy/bin/python \
  indy_loco/experiment/phase14_cubeai_conversion/run_pilot.py \
  --session indy_20160622_01 --fold 5 --overwrite
```

## Midsize deployment promotion

The six highlighted Phase-13 folds are now promoted to the midsize `AI` and
`deliverable3` branches with a velocity-only GRU graph. This avoids allocating
the unused 12,800-byte hidden-state output in Midsize. The promoted ABI is
`tcn64i8x6-gru64f32-p13-v3`, and calibration is 10,500 contiguous 40 ms bins.

Rebuild the exact firmware C graphs and GUI bundles with:

```bash
.venv-deploy/bin/python \
  indy_loco/experiment/phase14_cubeai_conversion/deploy_midsize.py
```

Large remains intentionally unchanged until the separate memlib deployment
step.
