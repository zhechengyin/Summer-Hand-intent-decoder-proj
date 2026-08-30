# Phase 15 — Large external-memory validation

This experiment measures the paper-facing Large result over all six sessions
and all five validation-selected Phase-13 folds. Large uses the same frozen
TCN+GRU checkpoint as Midsize plus a fold-specific residual bank queried by
`GRU hidden[49] + long context`.

Protocol:

- seven-minute calibration and continuous rolling windows from Phase 13;
- bank residuals and PCA fits from training bins only;
- retrieval hyperparameters selected from validation bins only;
- one final test evaluation per fold;
- 32D GRU PCA + 32D context PCA, INT8-rounded keys, FP16 residuals;
- exact PC KNN, which measures memory quality but not firmware IVF latency or
  approximate-search recall.

Run the complete check and experiment with:

```bash
.venv-deploy/bin/python \
  indy_loco/experiment/phase15_large_memory_validation/run.py \
  --validate-only --device cpu

.venv-deploy/bin/python \
  indy_loco/experiment/phase15_large_memory_validation/run.py \
  --device cpu --threads 4 --batch-size 512 --resume
```

All fold metrics, summaries, and PC `.memlib` artifacts are written beneath
`results/`. These `.memlib` files are reproducible PC evaluation archives, not
firmware-ready BCIMEM binaries.

Final result: **R² 0.7498 ± 0.0632** over 30 folds, versus **0.7411 ± 0.0656**
with the bank absent (`+0.0086`). See `TECHNICAL_REPORT.md` for the session
table, paired statistics, negative folds, and interpretation boundary.

## Firmware deployment check

The six GUI-deployed best-fold banks were separately packed into `BCIMEM1`
images and replayed with the CM7 IVF policy (256 clusters, 32 probes, INT8 dot
products, FP16 residuals). Across those six demonstration folds the mean was
0.794441 bank-ABSENT, 0.796320 firmware-IVF READY (`+0.001879`); IVF was only
0.000167 below exact KNN. This is a deployment-format check, not the paper
estimate, and one selected fold (`loco_20170301_05` fold 1) remains negative.
See `results/phase15_firmware_ivf_bestfolds.json`.
