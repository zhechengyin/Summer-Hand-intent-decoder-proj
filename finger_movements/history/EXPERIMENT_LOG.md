# FingerMovements Experiment Log

This file records what was tested, which evidence remains valid, and why the current model was selected.

## Evidence boundary

On 2026-08-10, the original UEA-derived conversion was found to contain deterministic adjacent-channel overlap. Every result produced from that conversion is invalid for comparison with the corrected official MATLAB data. The scripts and outputs remain only as provenance.

## Phase summary

| Date | Phase | Data source | Main result | Decision |
|---|---|---|---|---|
| 2026-07-30 | 1b | retired UEA | initial neural and linear baselines | invalidated |
| 2026-08-02–05 | 1c–1h | retired UEA | terminal features + Logistic selected; TEST BA 62.10% | invalidated |
| 2026-08-05 | A2 | retired UEA | CSSD + hierarchical LDA BA 59.81% | invalidated |
| 2026-08-07 | 2b | retired UEA | isolated/combined CSSD tests up to 63.41% | invalidated |
| 2026-08-10 | correction | official MATLAB | direct conversion replaced UEA representation | new evidence boundary |
| 2026-08-10 | corrected A2 | official MATLAB TRAIN | BA 85.03% | establish CSSD baseline |
| 2026-08-10 | corrected 2b | official MATLAB TRAIN | zero-phase BA 86.72% | retain best offline reference |
| 2026-08-10 | corrected Logistic | official MATLAB TRAIN | BA 78.58% | linear control only |
| 2026-08-10–11 | 2c | official MATLAB TRAIN | causal 400 ms BA 83.99% | promote deployable checkpoint |
| 2026-08-11 | 2d | official MATLAB TEST | frozen BA 77.05% | retrospective benchmark only |
| 2026-08-12 | 2e | official MATLAB TRAIN | ToeplitzLDA BA 84.50% | no stable promotion |
| 2026-08-12 | 2f | official MATLAB TRAIN | Riemannian BA 85.13% | no stable promotion |
| 2026-08-13 | firmware | frozen Phase 2c | C/Python predictions matched on 316 cases | retain C99 port |

## Retired Phase 1 and early Phase 2 results

Phase 1 compared small neural networks, hand-built feature models, classifiers, regularization, and terminal feature groups. It eventually selected terminal low-pass samples plus logistic regression. The final all-TRAIN fit and one-time TEST inference reported 68.89% development BA and 62.10% TEST BA.

The first CSSD experiments reported 59.81% (Phase A2) and up to 63.41% (Phase 2b). These unexpectedly weak results helped reveal a source-format problem. After the official MATLAB release was converted directly, adjacent UEA cases were shown to share channels deterministically. All numbers in this paragraph are therefore invalid-source history, not model evidence.

## Corrected Phase A2 — Paper-style CSSD + LDA

The corrected runner used official MATLAB TRAIN only and fitted every temporal filter, CSSD spatial filter, scaler, and classifier inside each fold. Across seeds 42–44 and five folds, mean OOF balanced accuracy was 85.03%, seed SD 1.27 points, and worst-seed BA 83.25%.

The pipeline used low-frequency BP and ERD/F2 spatial branches plus a BP-trend branch. Branch LDA scores were fused by a final LDA. This established that the paper-style spatial pipeline was materially stronger than the corrected terminal-logistic control.

## Corrected Phase 2b — Offline configuration ablation

The 36-combination ablation isolated covariance choice, per-trial trace normalization, F2 component count, and fusion method. The winner used:

- empirical CSSD covariance;
- per-trial trace normalization;
- one F2 component per class;
- LDA fusion.

It achieved 86.72% mean OOF BA, 0.68-point seed SD, and 86.09% worst-seed BA, improving all three seeds over corrected Phase A2. Because temporal filtering was zero-phase, this is the best offline reference but not the deployable model.

The corrected terminal-feature logistic control reached 78.58% mean BA, confirming that spatial CSSD structure contributed substantial predictive value.

## Phase 2c — Strictly causal model

The first horizon diagnostic accumulated causal history from 50 to 500 ms. Mean OOF BA rose from 50.62% at 50 ms to 82.93% at 500 ms. Extreme perturbations applied only after the prediction point changed current scores by exactly zero.

The streaming implementation then clarified timing: the history window ends at the current prediction point; it does not wait for future EEG. Causal filter state is carried between updates, and a ring buffer stores past filtered samples.

A final TRAIN-only sweep compared 200, 300, 400, and 500 ms past windows:

| Window | Mean OOF BA |
|---:|---:|
| 200 ms | 79.62% |
| 300 ms | 79.43% |
| 400 ms | 83.99% |
| 500 ms | 82.93% |

The 400 ms window improved mean BA, worst-seed BA, and seed stability over 500 ms. Bin sizes of 10, 20, 50, and 100 ms produced equivalent endpoint features, so 50 ms remained the firmware update interval.

The all-TRAIN checkpoint was fitted once on 316 cases and verified after reload. Its 89.89% fitting BA is descriptive only; 83.99% OOF BA is the TRAIN-only generalization estimate.

## Phase 2d — Frozen official TEST inference

The exact 400 ms checkpoint was applied to the corrected 100-case TEST without fitting, recalibration, threshold selection, or TEST-derived preprocessing.

| Metric | Result |
|---|---:|
| Accuracy | 77.00% |
| Balanced accuracy | 77.05% |
| Macro-F1 | 77.00% |
| Confusion matrix | `[[39, 10], [13, 38]]` |

The TEST set had been exposed during the invalid UEA phase, so this is a retrospective benchmark rather than a pristine blind evaluation. It did not change the checkpoint.

## Phase 2e — Lightweight linear alternatives

The same TRAIN-only folds compared the unchanged Phase 2c baseline with regularized CSSD, shrinkage LDA, their combination, block-Toeplitz LDA, and conditional fusion.

- Regularized CSSD: 84.10% mean BA.
- ToeplitzLDA: 84.50% mean BA.
- Baseline + Toeplitz nested fusion: 84.09% mean BA.

No candidate improved all seeds while also improving worst-seed performance and reducing variability. No checkpoint was promoted.

## Phase 2f — Low-dimensional Riemannian model

The Riemannian tangent-space candidate reached 85.13% mean BA and 84.17% worst-seed BA, compared with the 83.99% causal baseline. Seed changes were +1.25, -0.03, and +2.21 points for seeds 42–44. Because seed 43 did not improve and seed/fold variability increased, it failed the frozen promotion rule. No checkpoint was created.

## Firmware conversion

The active 400 ms checkpoint was converted to a self-contained float32 C99 streaming implementation. On all 316 TRAIN cases, Python and C produced identical class labels. Maximum score and probability errors were `1.335e-4` and `3.228e-5`; block and chunk processing also matched.

This validated numerical equivalence on stored epochs, not real-time behavior on continuous EEG or STM32 timing. Those remain deployment-validation tasks.

## Final decision

Model exploration closed with the Phase 2c 400 ms causal CSSD + hierarchical LDA checkpoint unchanged. Phase 2b is retained as the stronger offline upper reference; Phase 2e and 2f remain rejected alternatives. Official TEST must not be reused for tuning.
