# Frozen Indy 32-channel model

`checkpoint.pt` is the only retained model checkpoint.

The active code beside it is intentionally model-specific:

- `input_pipeline.py`: processed-session loading, normalization and windowing;
- `features.py`: raw counts plus causal EWMA;
- `model.py`: causal TCN+GRU architecture and metrics;
- `drift_detector.py`: active pre-January 60-second compatibility gate;
- `sampling.py`: frozen session-balanced sampling rule.

| Item | Value |
| --- | --- |
| SHA-256 | `2ee52c426ee43ba88cebe7c85dd8392f40f9e75748abe9bbf4e94093556363a5` |
| Size | 324,237 bytes |
| Parameters | 78,786 |
| Seed / epoch | 43 / 7 |
| Input | 32 counts + 32 causal-EWMA features |
| Sampling | session-balanced |
| Learning rate | 0.0009 |
| Weight decay | 0.060 |
| Dropout | 0.025 |

The checkpoint contains model weights, selected channels, target normalization,
training-derived feature variance floor, training/validation session lists and
the frozen model configuration. The reader-facing configuration is
`configs/indy_32ch.yaml`.

Evidence:

- December validation pooled R²: 0.560362.
- January locked pooled R²: 0.551146.
- January session-macro R²: 0.504789.
- January worst-session R²: -0.052402.

This is the frozen research candidate, not yet a deployment release. Promotion
still requires a validated label-free drift gate, int8 accuracy comparison and
measured STM32 memory/timing.

The detector is separate from the checkpoint and never changes decoder weights.
Its first candidate combines multi-month rate references with a
five-dimensional, full-covariance Gaussian KLD inspired by MINDFUL. The
authoritative detector protocol is `configs/indy_32ch_detector.yaml`; Phase 3a
results are not yet strong enough to freeze deployment thresholds.
