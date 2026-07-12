# tcn_gru_8ch — 8-channel STM32 decoder (deployment model of record)

The **deployable** decoder for the current hardware: **8 spike-detection channels**
→ 2D fingertip velocity, on an **STM32-class MCU**. Same dilated-TCN + bidirectional
-GRU architecture as [`../tcn_gru`](../tcn_gru) (`build_net`), shrunk to STM32 size
and trained on more sessions.

## Headline

| | value |
| --- | ---: |
| **TEST R²** (untouched test1) | **0.628** (r = 0.793) |
| fp32 size | 100 KB (~25.6k params) |
| **int8 size** | **~27 KB — lossless** (R² 0.628 unchanged) |
| channels | 8 (top-8 firing electrodes of train1-6, **fixed**) |
| metric | R² (coefficient of determination, avg of X,Y) |

Fits any STM32 (even F1-class flash) with large margin.

## Contents

- `config.py` — the exact recipe (preprocessing, architecture, 18 training sessions, channels).
- `evaluate.py` — reproduce the R² (fp32 + int8) on the fixed split.
- `train_and_save.py` — train and write `checkpoint.pt` (weights + channels + axes + norm).
- `checkpoint.pt` — trained weights and everything needed to run/quantize.

```bash
py models/tcn_gru_8ch/evaluate.py        # reproduce R²
py models/tcn_gru_8ch/train_and_save.py  # (re)train and save the checkpoint
```

## How this R² was reached (DAILY_LOG LOG-050..054)

- **More training data is the lever**: 6→9→12→18 sessions = 0.529→0.589→0.616→0.628.
  Cheap training tricks (correlation loss, augmentation, regularization) were a wash;
  only seed-ensembling added +0.022 at 3× cost.
- **int8 quantization is free** (100 KB → 27 KB, no R² loss).
- **Do NOT re-select channels** on more data — it overfits (0.628 → 0.502). Keep the
  top-8 firing electrodes chosen on the original 6 sessions. Learned/correlation
  channel selection also lose to firing-rate.
- The paper's **Bessel output filter is redundant here** — we low-pass the velocity
  *target* at 3 Hz before training, so predictions are already smooth.

## Caveats

- **Cross-session, zero-calibration** number (train on other sessions, test on a
  held-out day). A per-user calibrated (within-session) decoder would score higher.
- R² 0.628 uses a **bidirectional** GRU (peeks ahead within the 2 s window). A strict
  real-time **causal** model costs ~0.078 R². Decide based on the latency budget.

Reference/SOTA: Zhou, Sun, Basu, "Motor decoding for iBMI" (arXiv:2312.15889, NeuroBench).
