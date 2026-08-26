# Indy Deployment Models

This directory contains the two current self-contained deployment candidates.
Each model subdirectory contains its frozen `checkpoint.pt`, standalone
`model.py`, and streaming `runtime.py`. Large is intentionally absent until a
general Large checkpoint has been trained and validated.

## Shared signal path

Both models predict x/y fingertip velocity from 40 ms spike-count bins. The
input features always place raw counts first and their causal EWMA values
second. The EWMA uses `alpha = 0.1` and never reads a future bin.

Each session begins with a 60-second calibration period containing 1,500 bins.
Per-feature mean and population standard deviation are estimated only from
this prefix. The effective standard deviation is the larger of the local value
and the training-derived floor stored in the checkpoint. Calibration statistics
remain frozen for the rest of the session.

Both neural networks use the same causal pattern:

1. a pointwise `Conv1D`, followed by `LayerNorm` and `ReLU`;
2. four residual causal TCN blocks with kernel size 3 and dilations
   `1, 2, 4, 8`;
3. one unidirectional GRU layer;
4. a linear two-output velocity head.

The TCN receptive field is 31 bins, or 1.24 seconds. The model input window is
50 bins, or 2 seconds. Neither deployment model is bidirectional.

## Tiny

| Item | Value |
|---|---|
| Directory | `tiny/` |
| Physical input | 32 fixed channels selected by the checkpoint |
| Feature input | 32 raw counts + 32 causal EWMAs = 64 features |
| Pointwise/TCN width | 48 |
| GRU width | 48 |
| Parameters | 45,266 |
| Checkpoint validation | December pooled R² 0.5651 |
| Checkpoint SHA-256 | `5c8b375787ff93f90006df5f0cfea07303660928c7b69a84d4d75e1a368319ef` |

Tiny is the compact firmware candidate. Its runtime preserves the model's
original strict causal block protocol: post-calibration inference starts at bin
1500, each 50-bin block is initially zero, and the block is cleared every two
seconds. At timestep `t`, columns `0..t` contain observed normalized features
and the unseen suffix remains zero.

## Midsize

| Item | Value |
|---|---|
| Directory | `midsize/` |
| Physical input | All 96 channels in order `0..95` |
| Feature input | 96 raw counts + 96 causal EWMAs = 192 features |
| Pointwise/TCN width | 64 |
| GRU width | 64 |
| Parameters | 86,978 |
| Phase 6 validation | December pooled R² 0.7022 |
| Phase 9 rolling replay | December pooled R² 0.7526; January pooled R² 0.7277 |
| Checkpoint SHA-256 | `685ee659b56e40d2484d09b4d03bbdcb032856e772228fb0125c3703575e378a` |

Midsize is the primary real-time candidate. Its runtime implements the Phase 9
deployment policy. During calibration it retains the final 50 unnormalized
feature bins. At exactly 60 seconds, it freezes the calibration statistics,
normalizes bins `1450..1499`, and produces the first prediction from timestep
49. Every subsequent 40 ms bin advances a continuous stride-1 rolling window.
The window is never cleared, and every prediction uses only the current and
previous bins.

Phase 9 validated this rolling policy for Midsize only. Tiny therefore retains
its own original block contract rather than inheriting an untested policy.
