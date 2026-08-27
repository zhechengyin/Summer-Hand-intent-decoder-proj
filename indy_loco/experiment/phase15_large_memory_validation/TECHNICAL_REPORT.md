# Phase 15: Large external-memory 30-fold PC validation

## Conclusion

The Phase-13 Large model with a fold-specific GRU-hidden[49] residual bank
achieved **R² 0.7498 ± 0.0632** across all 30 validation-selected test folds.
The matched bank-ABSENT neural baseline was independently reproduced at
**0.7411 ± 0.0656**, giving a mean gain of **+0.0086 R²**.

All six session means improved and 26 of 30 individual folds improved. A paired
session-level bootstrap gives a 95% interval of **[+0.0066, +0.0111]** for the
mean gain. The exact paired Wilcoxon result over the six session means is
one-sided `p=0.015625` and two-sided `p=0.03125`.

This is a statistically consistent but practically modest improvement. It does
not reproduce the much larger gain previously seen when only one test-selected
fold per session was used. The 30-fold result is the appropriate paper-facing
estimate because checkpoint selection and memory tuning do not use test R².

## Session results

| Session | Bank ABSENT R² | Bank READY R² | Delta |
|---|---:|---:|---:|
| indy_20160622_01 | 0.8381 | 0.8481 | +0.0100 |
| indy_20160630_01 | 0.7152 | 0.7217 | +0.0065 |
| indy_20170131_02 | 0.7725 | 0.7792 | +0.0068 |
| loco_20170210_03 | 0.7163 | 0.7223 | +0.0060 |
| loco_20170215_02 | 0.6826 | 0.6970 | +0.0144 |
| loco_20170301_05 | 0.7221 | 0.7302 | +0.0081 |
| **30-fold macro** | **0.7411** | **0.7498** | **+0.0086** |

Subject-level results were 0.7753 to 0.7830 for Indy (`+0.0078`) and 0.7070
to 0.7165 for Loco (`+0.0095`).

Four folds decreased on test after validation-only tuning:

- indy_20160630_01 fold 5: `-0.0013`
- indy_20170131_02 fold 3: `-0.0007`
- loco_20170215_02 fold 1: `-0.0033`
- loco_20170301_05 fold 1: `-0.0096`

These negative folds are retained in the reported average; no fold was removed
or replaced after observing test performance.

## Validation contract

- Six sessions times five reach-level folds; each Phase-13 neural checkpoint
  was selected by validation loss before test evaluation.
- Seven-minute calibration, continuous causal EWMA, past-only 50-bin rolling
  windows, and output timestep 49 exactly match Phase 13.
- The residual bank, GRU PCA, and context PCA use training bins only.
- Retrieval neighbour count, temperature, and residual blend use validation
  bins only.
- Test data is used once for the bank-ABSENT versus bank-READY comparison.
- The query is GRU hidden[49] reduced from 64D to 32D plus long context reduced
  from 576D to 32D.
- Keys are rounded to INT8 and stored residuals are rounded to FP16 before the
  PC replay.

The bank-ABSENT replay matched the saved Phase-13 30-fold result to the script's
`2e-6` tolerance on every fold. All 30 exported PC memlibs passed schema, dtype,
shape, and entry-count checks.

## Interpretation boundary

The retrieval in this experiment is exact `cKDTree` KNN over INT8-rounded keys.
It validates memory quality on PC, but it is not a measurement of firmware IVF
recall, MCU search latency, or board-level CubeAI parity. The 30 generated
`.memlib` files use the self-describing `phase15_pc_memlib_v1` evaluation
schema and are deliberately not advertised as firmware-compatible BCIMEM
binaries.

The seven-minute calibration prefix and continuous causal context are
deployment-legal and label-free, but they are transductive within a session.
This experiment is within-session five-fold validation, not cross-session or
unseen-subject generalization.

## Reproducibility

- Runner: `run.py`
- Fold metrics: `results/phase15_large_memory_folds.csv`
- Session metrics: `results/phase15_large_memory_sessions.csv`
- Machine-readable summary: `results/phase15_large_memory_summary.json`
- Per-fold audit records: `results/by_fold/<session>/fold-<n>.json`
- PC memory banks: `results/memlibs/<session>/fold-<n>.memlib`

Re-run from the repository root with:

```bash
.venv-deploy/bin/python \
  indy_loco/experiment/phase15_large_memory_validation/run.py \
  --device cpu --threads 4 --batch-size 512 --resume
```
