# Frozen Indy 32-channel model

`checkpoint.pt` is the only retained model checkpoint.

The active code beside it is intentionally model-specific:

- `input_pipeline.py`: processed-session loading, normalization and windowing;
- `features.py`: raw counts plus causal EWMA;
- `model.py`: causal TCN+GRU architecture and metrics;
- `drift_detector.py`: first-layer raw-count compatibility checks;
- `decoder_state_detector.py`: second-layer hidden/output detector and the
  active two-layer gate wrapper;
- `runtime.py`: the only integrated gate-then-decode execution path;
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

The detector never changes decoder weights. Layer 1 combines multi-month rate
references with a five-dimensional, full-covariance Gaussian KLD inspired by
MINDFUL. Layer 2 runs inference during the gated 60-second prefix and vetoes
only when both GRU-hidden and absolute-output distributions exceed their
reference-only severe thresholds. Output-delta and 10-second temporal scores
remain diagnostic only.

Phase 3c caught both known negative-R² pre-January sessions and passed the other
31 across nine detector sensitivity variants. The implementation is therefore
integrated as a development candidate, not as prospectively validated behavior.

The final saved gate is fitted on 31 compatible development references. With
the retained active checkpoint it blocks the known October 13 failure but not
June 30, because that checkpoint had already trained on both sessions. The
stronger two-failure result belongs specifically to strict held-month
evaluation. The authoritative protocol and this limitation are recorded in
`configs/indy_32ch_detector.yaml`.

Offline runtime example:

```bash
python models/indy_32ch/runtime.py \
  --session indy_20161207_02 \
  --device cpu
```

The first 60 seconds are diagnostic warm-up only. `abstain` blocks output;
`warning` releases output by default and can be made blocking with
`--block-on-warning`.
