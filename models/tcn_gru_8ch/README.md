# tcn_gru_8ch — historical 8-channel STM32 baseline

> Status: retained for reproduction, not the current deployment model. Although
> the network is unidirectional, this checkpoint was trained with centered
> Gaussian input smoothing and therefore is not fully causal end to end. See
> `../../docs/STATUS.md` and `../../configs/indy_8ch.yaml`.

This is the historical decoder for the 8-channel hardware: **8 spike-detection
channels** → **multi-timescale input** (raw + causal EWMA, 16 features) →
dilated-TCN + a unidirectional GRU ([`../tcn_gru`](../tcn_gru) `build_net` with
`bidir=False`). The neural network itself is causal, but the saved preprocessing
uses a centered Gaussian and reads future bins.

## Historical result (not an end-to-end deployment claim)

| | value |
| --- | ---: |
| **TEST R²** (test1, historical protocol) | **≈ 0.63** (3-seed) |
| network lookahead | **0 ms** |
| preprocessing lookahead | centered Gaussian, approximately **160 ms** at 40 ms/bin |
| compute | ~5.6 ms/pred, 1 CPU core |
| int8 size | **~74 KB — lossless** |
| input | 8 channels × {raw, EWMA} = 16 features |
| channels | top-8 firing electrodes of train1-6, **fixed** |
| training | 24 indy sessions, 40 ms bins |

Multi-timescale input beat single-scale on **both eval and test** (LOG-068) — the
first model-side lever to beat the 0.606 causal ceiling; it is adopted.

> ⚠️ **Methodology caveat (LOG-073).** The earlier **0.646** headline was
> **test-selected**: the EWMA α was tuned on `test1` R² (leakage). The eval-valid
> pick (α=0.1) scores **0.630** on test; α barely matters (within noise). Treat
> **≈ 0.63** only as a historical model-comparison result. Also, `test1` has
> now been read across ~25
> experiments, so it is **no longer a truly untouched test set** — an unbiased final
> headline needs a freshly reserved session, a causal preprocessing pipeline
> frozen in advance, and a single final score.
> Rule: select configs on **eval**, read test **once**; never promote a config
> because its test score is higher.

**Offline references only (NOT deployable):** a *bidirectional* GRU scores 0.677 but
needs the whole future; *bounded lookahead* gives 0.619 @ 80 ms / 0.623 @ 200 ms —
at 40 ms/bin that latency is too high for closed-loop (LOG-062).

`checkpoint.pt` = the unidirectional + multiscale weights (75,714 params).
`evaluate.py` / `export_int8.py` reproduce and quantize the historical artifact.

## Contents

- `config.py` — the exact recipe (preprocessing, causal architecture, 24 sessions, channels).
- `evaluate.py` — reproduce R² (fp32 + int8) on the fixed split.
- `train_and_save.py` — train and write `checkpoint.pt`.
- `export_int8.py` — quantize the saved checkpoint and report R² kept.

```bash
py models/tcn_gru_8ch/evaluate.py        # reproduce R²
py models/tcn_gru_8ch/train_and_save.py  # reproduce the historical checkpoint
```

## What we learned (LOG-050..065)

- **More training data is the one lever that worked**: 6→24 sessions lifted R²
  substantially. Beyond ~24 nearby sessions it plateaus; *distant* sessions add
  drift and slightly hurt (0.606→0.600, LOG-065).
- **Everything else is a dead end** (for both causal and bidir): architecture,
  depth, width/capacity (plateaus ~220 KB), correlation loss, augmentation,
  regularization, output smoothing (Bessel/EMA — redundant, our target is 3 Hz
  low-passed), and overlapping-window binning (40 ms boxcar is best).
- **Channels**: firing-rate top-8 on the base-6 sessions is best (0.655 in the
  bidir frame). Learned / low-freq / fft / re-selection all lose or tie (LOG-063).
- **int8 quantization is free** (no R² loss).
- A unidirectional network costs ~0.07 R² versus the bidirectional upper bound
  (0.677 → 0.606). A truly deployable successor must also replace the centered
  input filter with causal features.

## Where more R² could come from (not model-side)

More channels (hardware), sessions closer in time to the user, per-user
calibration on top of the pool, or richer signal (broadband). All are data/hardware
levers — the decoder itself is well-tuned and near its ceiling for this input.

Reference/SOTA: Zhou, Sun, Basu, "Motor decoding for iBMI" (arXiv:2312.15889, NeuroBench).
