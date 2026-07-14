# tcn_gru_8ch — 8-channel STM32 decoder (deployment model of record)

The **deployable** decoder for the current hardware: **8 spike-detection channels**
→ **multi-timescale input** (raw + causal EWMA, 16 features) → dilated-TCN +
**strictly-causal (unidirectional) GRU** ([`../tcn_gru`](../tcn_gru) `build_net`
with `bidir=False`), STM32-sized, trained on 24 sessions.

## Headline (deployable = strictly causal + multiscale)

| | value |
| --- | ---: |
| **TEST R²** (untouched test1, causal) | **0.633** single / **0.646** 3-seed ensemble |
| latency | **0 ms lookahead** (real-time) |
| compute | ~5.6 ms/pred, 1 CPU core |
| int8 size | **~74 KB — lossless** (0.633 → 0.633) |
| input | 8 channels × {raw, EWMA α=0.2} = 16 features |
| channels | top-8 firing electrodes of train1-6, **fixed** |
| training | 24 indy sessions, 40 ms bins |

Multi-timescale input added **+0.027 (single) / +0.028 (ensemble)** over single-scale
(LOG-068/070/071) — the first model-side lever to beat the 0.606 causal ceiling.

**Offline references only (NOT deployable):** a *bidirectional* GRU scores 0.677 but
needs the whole future; *bounded lookahead* gives 0.619 @ 80 ms / 0.623 @ 200 ms —
at 40 ms/bin that latency is too high for closed-loop (LOG-062).

`checkpoint.pt` = the causal + multiscale weights (75,714 params). `evaluate.py` /
`export_int8.py` reproduce and quantize it.

## Contents

- `config.py` — the exact recipe (preprocessing, causal architecture, 24 sessions, channels).
- `evaluate.py` — reproduce R² (fp32 + int8) on the fixed split.
- `train_and_save.py` — train and write `checkpoint.pt`.
- `export_int8.py` — quantize the saved checkpoint and report R² kept.

```bash
py models/tcn_gru_8ch/evaluate.py        # reproduce R²
py models/tcn_gru_8ch/train_and_save.py  # (re)train + save the (now causal) checkpoint
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
- **Causality costs ~0.07 R²** (0.677 bidir → 0.606 causal); lookahead buys back
  little and plateaus fast, so strictly causal is the right deployable choice.

## Where more R² could come from (not model-side)

More channels (hardware), sessions closer in time to the user, per-user
calibration on top of the pool, or richer signal (broadband). All are data/hardware
levers — the decoder itself is well-tuned and near its ceiling for this input.

Reference/SOTA: Zhou, Sun, Basu, "Motor decoding for iBMI" (arXiv:2312.15889, NeuroBench).
