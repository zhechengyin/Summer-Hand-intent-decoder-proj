# Lightweight decoder architecture experiment

## Goal

Determine whether the current TCN+GRU is still the best 8-channel decoder, and
find the strongest architecture that can run with acceptable real-time latency
on an STM32-class device.

## Keep fixed

- Training data: current 24-session pool.
- Validation/test: current `eval1` and untouched `test1`.
- Input: fixed firing-rate channels `[26, 51, 53, 66, 71, 73, 75, 94]`.
- Preprocessing: 40 ms bins, 2 s history, 3 Hz velocity-target low-pass.
- Output: 2D fingertip velocity.
- Training seed, epochs, optimizer, augmentation, and early-selection procedure.

Do not retune channels or preprocessing separately for each architecture.

## Models

Run these in order and stop testing clearly inferior variants:

1. **Current wide bidirectional TCN+GRU** — reference result, R² ≈ 0.67.
2. **Wide causal TCN** — residual dilated Conv1D blocks and linear output head.
3. **Depthwise-separable causal TCN** — MCU-efficient version of model 2.
4. **GRU only** — tests whether the TCN front end is necessary.
5. **Causal TCN+GRU** — tests the hybrid without future leakage.
6. **Bounded-lookahead TCN+GRU** — only if the causal models lose badly; permit
   80–200 ms of future context rather than the full 2 s window.

Use approximately comparable parameter counts where practical. Do not perform a
large hyperparameter sweep: one small and one wide configuration is enough for
any architecture that looks competitive.

## Report only

| Metric | Why |
| --- | --- |
| **Test R²** | Primary decoding-quality metric. |
| **Approximate int8 size** | Confirms STM32 storage feasibility. |
| **Inference latency** | Determines real-time usability. |

Also label each model as causal or state its exact lookahead. Pearson `r`, MAE,
RMSE, per-axis tables, and extensive training diagnostics are unnecessary unless
two models are effectively tied.

## Decision rule

- Prefer the highest test R² among models that fit the device and meet latency.
- If two models are within 0.01 R², choose the causal model with lower latency.
- A bounded-lookahead model is acceptable only if its gain clearly justifies the
  added delay.
- Keep the current TCN+GRU if no alternative improves the real deployment
  tradeoff.
