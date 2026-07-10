# Current best model

The frontier decoder of record: **per-electrode multiunit spike rates → 2D
fingertip velocity**, using a dilated causal **TCN + bidirectional GRU** with a
per-timestep linear head (`frontier.core.build_net`).

## Headline numbers (Pearson r, held-out sessions never seen in training)

| Setting | Held-out mean r |
| --- | ---: |
| Full 96 electrodes (cross-session) | **0.87** |
| 8 electrodes, top-8 by firing (hardware limit) | 0.76 |
| Random 8 electrodes (floor) | 0.69 |

Model: ~192k params, **0.77 MB** (fp32). Real-time: ~6 ms/prediction
bidirectional, ~3.7 ms causal, on 1 CPU core.

## Exact configuration

See [`config.py`](config.py) — the single source of truth:

- **Preprocessing:** 40 ms bins (25 Hz), 2 s windows, per-electrode multiunit
  spike counts (96 session-consistent channels), per-session channel z-score,
  3 Hz low-pass on finger position before differentiating, σ=1 bin firing-rate
  Gaussian smoothing. Target = 2D top-2 velocity-variance movement axes.
- **Architecture:** spatial 1×1 conv → BN → GELU; TCN dilations [1,2,4,8,16]
  (F=64, residual, causal); 2-layer bidirectional GRU (H=64); linear head.
  Activation ReLU in conv/TCN. Dropout 0.3.
- **Training:** AdamW lr 1e-3, wd 1e-3, cosine schedule, 60 epochs, batch 32,
  input Gaussian noise 0.1 + per-sample channel dropout 0.1.

## Reproduce / use

```bash
# reproduce the held-out 0.87 metric (trains + evaluates, cross-session)
py frontier/crosssession.py

# train on all training sessions and save a deployable checkpoint here
py frontier/best_model/train_and_save.py   # -> checkpoint.pt
```

`checkpoint.pt` stores `{state_dict, config, axes, y_mean, y_std}` so a model can
be rebuilt with `frontier.core.build_net(config, n_ch)` and de-normalized.

## Provenance

Established over LOG-026→033 (cross-session generalisation + tuning) and
LOG-042 (8-channel hardware constraint). Does **not** transfer across subjects
(indy→loco collapses, LOG-027) — needs per-subject calibration.
