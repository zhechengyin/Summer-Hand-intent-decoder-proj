# Indy 32-channel model

Two checkpoints are retained. `64x64checkpoint.pt` is the protected,
detector-compatible baseline used by `runtime.py`.
`48x48checkpoint.pt` is the completed firmware candidate; it must not replace
the integrated baseline until its 48-state Layer-2 detector is refitted and
validated.

The active code beside it is intentionally model-specific:

- `input_pipeline.py`: processed-session loading, normalization and windowing;
- `features.py`: raw counts plus causal EWMA;
- `model.py`: causal TCN+GRU architecture and metrics;
- `drift_detector.py`: first-layer raw-count compatibility checks;
- `decoder_state_detector.py`: second-layer hidden/output detector and the
  active two-layer gate wrapper;
- `runtime.py`: the only integrated gate-then-decode execution path;
- `sampling.py`: frozen session-balanced sampling rule.

| Item | Integrated 64/64 | Firmware candidate 48/48 |
| --- | --- | --- |
| File | `64x64checkpoint.pt` | `48x48checkpoint.pt` |
| SHA-256 | `2ee52c…56363a5` | `5c8b375…368319ef` |
| Size | 324,237 bytes | 199,733 bytes |
| Parameters | 78,786 | 45,266 |
| Seed / selected epoch | 43 / 7 | 43 / 10 |
| Training budget | 20 epochs | 20 epochs |
| December pooled loss | 0.480166 | 0.470492 |
| December pooled R² | 0.560362 | 0.565134 |
| December macro R² | 0.570223 | 0.575004 |
| December worst-session R² | 0.314350 | 0.346125 |

Both use 32 counts plus 32 causal-EWMA features, session-balanced sampling,
learning rate 0.0009, weight decay 0.060 and dropout 0.025.

The checkpoint contains model weights, selected channels, target normalization,
training-derived feature variance floor, training/validation session lists and
the frozen model configuration. The reader-facing configuration is
`configs/indy_32ch.yaml`.

Evidence:

- December validation pooled R²: 0.560362.
- January locked pooled R²: 0.551146.
- January session-macro R²: 0.504789.
- January worst-session R²: -0.052402.

Neither file is yet a deployment release. Promotion still requires a validated
48-state label-free drift gate, streaming/int8 equivalence and measured STM32
memory and timing.

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

## Baseline protection during Phase 4

Phase 4a nominated a 48-TCN-filter/48-GRU-hidden architecture with 45,266
parameters, versus 78,786 for the baseline. Phase 4b then completed five seeds
and five held-month folds per architecture. The 48/48 architecture passed all
four predeclared non-inferiority checks:

- mean session-macro R² delta: -0.005919, limit -0.010;
- mean session-q10 R² delta: +0.003694, limit -0.020;
- mean worst-session R² delta: -0.014631, limit -0.020;
- worst held-month macro R² delta: -0.018616, limit -0.020.

It is therefore a valid firmware architecture candidate, not a five-seed
accuracy winner. The subsequent fixed build completed on 2026-07-28: seed 43,
20 training epochs and minimum December validation loss selected epoch 10.
Only the 29 training sessions updated weights; December was inference-only and
January was never opened. Its epoch-7 CPU result reproduced the matching
Phase-4b cell exactly. The completed builder is archived at
`history/phase4/train_48x48_checkpoint.py`.
