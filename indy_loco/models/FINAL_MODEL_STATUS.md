# Final Phase-13 model conclusion

## Final decision

The final Python checkpoint set is **Phase 13 Round 3**. All 30 checkpoints were
selected by minimum validation loss and evaluated on test targets only after
selection. They implement seven-minute unlabeled calibration, continuous
session-level causal EWMA, rolling 50-bin past windows, and prediction from GRU
timestep 49.

The paper-facing Midsize result is:

\[
R^2 = \mathbf{0.7411 \pm 0.0656}
\]

This is the macro mean and standard deviation over all 30 folds. Because every
session contributes exactly five folds, the 30-fold mean is also the unweighted
mean of the six session means.

## Six-session cross-validation result

| Session | Five-fold test R² mean ± SD | Best test fold | Best-fold R² |
|---|---:|---:|---:|
| `indy_20160622_01` | 0.8381 ± 0.0214 | 5 | 0.8634 |
| `indy_20160630_01` | 0.7152 ± 0.0172 | 4 | 0.7358 |
| `indy_20170131_02` | 0.7725 ± 0.0562 | 4 | 0.8503 |
| `loco_20170210_03` | 0.7163 ± 0.0450 | 5 | 0.7878 |
| `loco_20170215_02` | 0.6826 ± 0.0431 | 5 | 0.7269 |
| `loco_20170301_05` | 0.7221 ± 0.0638 | 1 | 0.8025 |
| **All 30 folds** | **0.7411 ± 0.0656** | — | range 0.6260–0.8634 |

The best-fold column is descriptive and supports convenient future CubeAI or
board selection. It is not the result to place in the paper's primary accuracy
table.

## Alignment with the benchmark paper

The reporting unit now matches the paper-facing benchmark convention used in
this project: the same six Indy/Loco sessions, five folds per session, one
session-level mean per five folds, and a final aggregate across 30 folds. The
model architecture, 40 ms preprocessing, channel restriction, and deployment
calibration policy are still project-specific, so this is protocol-aligned
reporting rather than an exact reproduction of the paper's model pipeline.

## Midsize versus Large

| Tier | Neural folds ready | Compatible memory banks | Current paper R² | Status |
|---|---:|---:|---:|---|
| Midsize | 30/30 | not applicable | **0.7411 ± 0.0656** | final Python checkpoints; CubeAI pending |
| Large | 30/30 | 0/30 | not yet reportable | memory rebuild, five-fold replay, and CubeAI pending |

Large uses the same neural weights as Midsize and adds residual retrieval. The
Phase-12 evidence supports GRU hidden[49] over Encoder[49] as the query
representation, but those old results used older checkpoints. They cannot be
combined numerically with the Phase-13 Midsize mean. A valid Large paper number
requires one train-only bank per fold, validation-only retrieval tuning, and one
held-out test score per fold under the Phase-13 preprocessing contract.

## What is and is not final

- Final: checkpoint weights, five folds per session, metric definition,
  seven-minute preprocessing contract, best-fold markers, hashes, and Python
  model definition.
- Not run in this phase: ONNX/H5 export, X-CUBE-AI analyze/validate/generate,
  generated C, `.aibundle`, PC generated-C replay, or STM32 board testing.
- Pending for Large: 30 compatible GRU residual-memory banks and their
  cross-validated corrected R².

Source metrics are in
`../experiment/phase13_deployment_validation/results/rolling_retrain/phase7_all/phase13_round3_folds.csv`
and `phase13_round3_summary.csv` in the same directory.
