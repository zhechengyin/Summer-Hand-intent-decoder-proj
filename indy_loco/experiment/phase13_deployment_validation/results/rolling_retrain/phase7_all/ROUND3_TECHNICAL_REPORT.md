# Phase 13 Round 3: seven-minute rolling-window retraining

## Conclusion

Deployment-aligned retraining recovered the rolling/calibration mismatch and
improved beyond the matched reach-local reference. Across 30 validation-selected
folds, mean R² increased from **0.6728** with the old Phase-7 weights under
seven-minute rolling preprocessing to **0.7411** after retraining, a gain of
**+0.0684**. The retrained result is also **+0.0322** above the matched
reach-local score of 0.7089.

This is an unbiased five-fold mean with respect to checkpoint selection: each
checkpoint was selected using validation loss, saved, and only then evaluated
on its test targets. It is not a best-test-fold selection result and no model
has been promoted into `models/`, firmware, or GUI.

## Matched decomposition

| Numeric path on identical post-7-minute test bins | 30-fold macro R² | Delta |
|---|---:|---:|
| Phase-7 reach-local | 0.7089 | reference |
| Phase-7 weights, continuous rolling, training normalization | 0.7039 | −0.0050 vs reach-local |
| Phase-7 weights, continuous rolling, 7-minute calibration | 0.6728 | −0.0312 vs prior row |
| Retrained weights, continuous rolling, 7-minute calibration | **0.7411** | **+0.0684 vs old weights** |

The previously reported `−0.0228` came from the six selected best folds on a
different evaluation mask. It must not be numerically substituted into this
30-fold post-seven-minute comparison. Under the new matched protocol, the pure
rolling-window difference is `−0.0050`; the larger old-weight loss comes from
changing training normalization to seven-minute calibration (`−0.0312`). The
new training contract adapts to both differences.

## Session-level results

| Session | Reach-local | Old 7-min rolling | Retrained 7-min rolling | Retraining gain | Net vs reach-local |
|---|---:|---:|---:|---:|---:|
| indy_20160622_01 | 0.8104 | 0.7715 | **0.8381** | +0.0666 | +0.0277 |
| indy_20160630_01 | 0.6862 | 0.6292 | **0.7152** | +0.0859 | +0.0290 |
| indy_20170131_02 | 0.7268 | 0.6620 | **0.7725** | +0.1105 | +0.0457 |
| loco_20170210_03 | 0.6783 | 0.6644 | **0.7163** | +0.0519 | +0.0380 |
| loco_20170215_02 | 0.6509 | 0.6160 | **0.6826** | +0.0666 | +0.0317 |
| loco_20170301_05 | 0.7009 | 0.6934 | **0.7221** | +0.0287 | +0.0212 |
| Six-session / 30-fold macro | 0.7089 | 0.6728 | **0.7411** | **+0.0684** | **+0.0322** |

All six session-average retraining gains are positive. An exact two-sided
Wilcoxon signed-rank test over the six independent session means gives
`p=0.03125`. At fold level, 29/30 retraining gains are positive; the one negative
fold is `−0.0005`. Because folds from the same session are correlated, the
session-level test is the appropriate primary significance check.

The retrained model beats the matched reach-local score in 28/30 folds and all
six session means. The remaining two fold differences are small (`−0.0081` and
`−0.0023`).

## Weight-change audit

The default experiment warm-started each matching Phase-7 fold, trained all
weights, used `3e-4` for GRU/head, and used one quarter of that rate for the
encoder/TCN. Mean relative L2 changes across the 30 selected checkpoints were:

| Parameter group | Mean relative L2 change | Median |
|---|---:|---:|
| Encoder + TCN | 7.50% | 6.92% |
| GRU | **17.94%** | **16.57%** |
| Output head | 16.32% | 17.13% |

The GRU changed about 2.4 times as much as the encoder/TCN, consistent with the
intended hidden-state adaptation. This does **not** prove that GRU changes alone
cause the improvement because encoder/TCN weights were also updated. The script
supports a controlled `--train-scope gru-head` run to test that narrower causal
hypothesis.

## Training and validation contract

- Same Phase-7 five reach-level folds and fold seed 43.
- Loco channel selection remains training-reach-only for every fold.
- Input counts and velocity were checked element-for-element against the GUI
  deployment arrays before training.
- Continuous causal EWMA is built over the session and never reset at reach
  boundaries.
- Each example is the exact past-only 50-bin window ending at the predicted bin.
- Calibration mean/std use only the unlabeled first 10,500 bins (seven minutes).
- Target normalization uses training bins after calibration only.
- Loss is applied only to timestep 49, matching the firmware output timestep.
- Validation normalized MSE selects the checkpoint; test targets are evaluated
  only after the selected checkpoint is atomically saved.
- Early stopping patience is six non-improving epochs.

## Interpretation and limitations

The result supports retraining for the deployed temporal contract rather than
reusing reach-local weights. The overall 0.7411 is close to the previously
quoted 0.7461 best-fold deployment reference, but they are not the same metric:
0.7411 is a 30-fold macro mean on post-seven-minute bins, while 0.7461 is the
mean of six test-selected folds under the original selection report.

The calibration prefix is deployment-legal and unlabeled, but it can contain
input reaches assigned to any fold. That is intentional transductive
normalization matching the live session calibration policy; no labels or
targets from that prefix are used to choose channels, fit target scaling, train
weights, or select checkpoints.

These are FP32 PyTorch results. Before promotion, selected checkpoints still
require CubeAI conversion and PC/on-board numeric replay. The experiment also
does not yet isolate GRU-only tuning, compare seven minutes against other
calibration durations after retraining, or test genuinely unseen users.

## Recommended next experiment

Run the same 30-fold protocol with `--train-scope gru-head` and compare it with
`phase7_all`. If GRU/head-only tuning preserves most of the +0.0684 gain, it is
a cleaner and cheaper deployment-alignment method. If it loses materially,
retain all-weight tuning and treat the improvement as joint encoder/TCN/GRU
adaptation.

## Reproducibility

- Training script: `../../../run_rolling_retrain.py` relative to this report.
- Fold results: `phase13_round3_folds.csv`.
- Session and overall summary: `phase13_round3_summary.csv`.
- Epoch history: `phase13_round3_epochs.csv`.
- Full metrics and protocol: `phase13_round3_metrics.json`.
- Validation-selected checkpoints: `checkpoints/` (30 files).
- Resume state: `.state.json`.
