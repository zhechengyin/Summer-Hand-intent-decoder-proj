# tcn_gru_8ch — 8-channel STM32 decoder (deployment model of record)

The **deployable** decoder for the current hardware: **8 spike-detection channels**
→ 2D fingertip velocity, on an **STM32-class MCU**. Same dilated-TCN + bidirectional
-GRU architecture as [`../tcn_gru`](../tcn_gru) (`build_net`), shrunk to STM32 size
and trained on more sessions.

## Headline

| | value |
| --- | ---: |
| **TEST R²** (untouched test1) | **≈0.67** (0.668 checkpoint / 0.677 harness) |
| fp32 size | 392 KB (~100k params) |
| **int8 size** | **~98 KB — lossless** (R² unchanged) |
| channels | 8 (top-8 firing electrodes of train1-6, **fixed**) |
| training | 24 indy sessions |
| metric | R² (coefficient of determination, avg of X,Y) |

Fits STM32 F4/H7-class flash comfortably. For the tightest flash budgets, the
smaller **F32/H32 'small' variant is ~25 KB int8 at R² 0.655** (set `MODEL` in
`config.py` back to `F=32, H=32`). A single 'wide' model (0.677) already matches a
3-seed ensemble (0.675), so no ensemble is needed.

## Contents

- `config.py` — the exact recipe (preprocessing, architecture, 18 training sessions, channels).
- `evaluate.py` — reproduce the R² (fp32 + int8) on the fixed split.
- `train_and_save.py` — train and write `checkpoint.pt` (weights + channels + axes + norm).
- `checkpoint.pt` — trained weights and everything needed to run/quantize.

```bash
py models/tcn_gru_8ch/evaluate.py        # reproduce R²
py models/tcn_gru_8ch/train_and_save.py  # (re)train and save the checkpoint
```

## How this R² was reached (DAILY_LOG LOG-050..056)

- **More training data is the main lever**: 6→9→12→18→24 sessions =
  0.529→0.589→0.616→0.628→0.655 (fixed small model). Cheap training tricks
  (correlation loss, augmentation, regularization) were a wash.
- **More data then unlocks more capacity**: at 24 sessions the larger 'wide'
  model (F64/H64) reaches **0.677**, where at 6 sessions it overfit. This single
  model matches a 3-seed ensemble (0.675).
- **int8 quantization is free** (no R² loss; 'small' 100→27 KB, 'wide' 392→98 KB).
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
