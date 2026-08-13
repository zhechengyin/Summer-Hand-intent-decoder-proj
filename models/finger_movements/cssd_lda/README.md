# FingerMovements causal CSSD + hierarchical LDA

This is the active Phase 2c model. It predicts left/right movement from the
400 ms EEG interval ending at the current point A. The implementation is
strictly causal: temporal filters run left-to-right, samples after A are never
used, and the streaming API carries IIR state plus a rolling 400 ms buffer.

## Frozen baseline configuration

- 28 EEG channels sampled at 100 Hz;
- 400 ms feature window (40 samples);
- 100 ms causal filter pre-roll after a cold reset;
- 50 ms streaming update (5 new samples);
- fourth-order causal 0--7 Hz BP filter;
- fourth-order causal 10--33 Hz ERD filter;
- empirical CSSD covariance with per-trial trace normalization;
- one BP and one ERD spatial pattern per class;
- BP, ERD, and BP-trend branch LDAs followed by LDA fusion.

TRAIN-only repeated five-fold validation across seeds 42/43/44 produced
83.99% mean OOF balanced accuracy, 0.54 percentage-point seed standard
deviation, and 83.25% worst-seed balanced accuracy.

## Checkpoint

The active checkpoint is:

```text
checkpoints/finger_movements_cssd_lda_phase2c_causal_400ms.npz
```

Its adjacent `.metrics.json` records the exact checkpoint hash, source-data
hash, development evidence, full-TRAIN fit diagnostics, reload verification,
and streaming-equivalence verification.

Checkpoint SHA-256:

```text
87b84cc2c8baf9efdc1ccf37ad28f5f58ad13c4db2a8f8a273fe73fce9956101
```

The apparent all-TRAIN balanced accuracy is 89.89%. It is only a fit
diagnostic; the frozen generalization estimate remains 83.99% mean OOF BA.

Phase 2d subsequently evaluated this exact checkpoint on the corrected
official TEST using pure inference: 77.00% accuracy, 77.05% balanced accuracy,
and 77.00% macro-F1 over 100 cases. That retrospective result is archived at
`history/finger_movements/results/phase2d_official_test_400ms/`; it did not
change the Phase 2c model or checkpoint.

Phase 2e then compared regularized CSSD, shrinkage LDA, their combination,
ToeplitzLDA, and conditional baseline/Toeplitz fusion on the same TRAIN-only
folds. ToeplitzLDA improved mean BA to 84.50% but failed the stability criteria:
seed 44 decreased, worst-seed BA did not improve, and variability increased.
No Phase 2e method replaced this checkpoint.

Rebuild it from official TRAIN only:

```bash
python models/finger_movements/cssd_lda/train_checkpoint.py
```

The entry point refuses any data path containing `test`. Apparent metrics on
the 316 fitting cases are diagnostics, not held-out performance.

## Deployment boundary

The model is causal, but the official dataset contains isolated 500 ms epochs
rather than continuous EEG. The first 100 ms supplies causal filter pre-roll;
the selected feature ring is the following 400 ms. After cold reset, the first
validated output therefore occurs at 500 ms; subsequent outputs update every
50 ms. Real continuous recordings are still required to validate long-running
filter state, overlapping windows, and rest/no-intent behavior.

The previous 500 ms causal model/checkpoint and the zero-phase Phase 2b
reference are preserved under `history/finger_movements/models/`.

## Firmware C implementation

The frozen checkpoint now has a self-contained C99 streaming port under
`firmware/`. It includes generated float32 parameters, a no-allocation causal
SOS/ring-buffer/LDA implementation, an STM32-style integration example, and a
host validation tool. All 316 official TRAIN cases produced identical Python
and C class labels; the maximum score error was `1.335e-4`.

See `firmware/README.md` for the exact channel/data contract, memory estimate,
integration procedure, and remaining target-hardware validation boundary.
