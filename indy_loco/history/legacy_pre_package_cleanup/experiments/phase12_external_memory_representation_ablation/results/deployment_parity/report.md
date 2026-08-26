# Phase 12 deployment-parity GRU memory A/B

## Decision summary

With each GUI-selected checkpoint and its selected-fold held-out reaches, the
matched continuous rolling replay improves from **0.6224** with the bank
`ABSENT` to **0.7396** with the GRU residual bank `READY`. The paired mean
uplift is **+0.1171 mean R²** (six-session bootstrap 95% CI **+0.0626 to
+0.1722**). All six sessions improve; the exact one-sided Wilcoxon and sign
tests both give **p = 0.015625**.

| Session | Selected fold | ABSENT | GRU READY | Uplift | Reach-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| indy_20160622_01 | 5 | 0.7311 | 0.8115 | +0.0804 | [+0.0605, +0.1019] |
| indy_20160630_01 | 4 | 0.5377 | 0.7016 | +0.1639 | [+0.1297, +0.1898] |
| indy_20170131_02 | 4 | 0.5660 | 0.7838 | +0.2178 | [+0.1801, +0.2609] |
| loco_20170210_03 | 5 | 0.6421 | 0.6982 | +0.0560 | [+0.0305, +0.0803] |
| loco_20170215_02 | 4 | 0.5133 | 0.6751 | +0.1618 | [+0.0961, +0.2322] |
| loco_20170301_05 | 1 | 0.7444 | 0.7672 | +0.0228 | [+0.0034, +0.0408] |

## What this resolves

The GUI mean **0.7461** is still the mean `selection_test_r2_mean`: it was
produced by Phase-7 chunked inference and by choosing the highest-test-R² fold.
It is not the bank-ABSENT score under the deployment rolling pipeline. The
matched rolling comparison is **0.6224 → 0.7396**. GRU memory therefore
recovers most of the rolling-policy gap, but it is not valid to describe the
result as `0.7461 + 0.1171`.

## Protocol and limitations

- Each bank and both PCA fits use selected-fold train reaches only.
- K, temperature, and blend are selected on validation reaches only; test
  reaches are evaluated once.
- The replay uses the saved deployment candidate, 60-second calibration,
  continuous EWMA, 50-bin rolling input, int8 64-D keys, and FP16 residuals.
- `ABSENT` is asserted against the existing per-session deployment replay to
  within 2e-6 for R² and MSE.
- Checkpoint/fold selection used test R², so absolute scores remain
  selection-biased and are not an unbiased generalization estimate.
- Retrieval here is exact PC cKDTree search over quantized keys. The generated
  `.memlib` archives are experimental and are not yet compatible with the
  firmware BCIMEM/IVF ABI; MCU recall and latency still require a firmware
  implementation and hardware replay.
