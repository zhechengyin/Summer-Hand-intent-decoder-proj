# research/ — active R² iterations

Active experiments to raise 8-channel velocity-decoding **R²** under the STM32
deployment constraints. Models of record live in [`../models/`](../models);
archived experiments in [`../legacy/`](../legacy). Everything here reuses the
fixed train/eval/test split and data pipeline from
[`../models/tcn_gru/evaluate.py`](../models/tcn_gru/evaluate.py) so numbers stay
comparable.

## Files

- `harness.py` — shared harness. `prep(nch)` loads the split (optionally top-`nch`
  by firing rate); `run(data, cfg, ...)` trains with early-stopping on eval and
  reports **both Pearson r and R²**, with swappable hooks: `build` (architecture),
  `loss_fn`, `post` (output filter), `seeds` (ensembling), `ret_preds` (raw
  predictions for offline filter sweeps). Run directly for the baseline.
- `iter1_cheap_wins.py` — correlation-aligned loss, ensembling, causal output EMA.
- `iter2_tiny_bessel.py` — STM32 shrink sweep + Bessel output filter (paper trick).

## Findings so far

| setting | TEST R² | size |
| --- | ---: | ---: |
| base TCN+GRU, 96 ch | 0.741 | 0.77 MB |
| base TCN+GRU, 8 ch | 0.548 | 0.75 MB |
| **`small`, 8 ch** (F32/H32/L1) | **0.529** | **100 KB** |
| `small` causal (real-time), 8 ch | 0.451 | 75 KB |

- Shrinking to ~100 KB costs only −0.019 R² → **size is not the bottleneck**.
- The Bessel output filter is **redundant** here (we low-pass the velocity target
  at 3 Hz before training). See DAILY_LOG LOG-050.
- **Causality costs ~0.078 R²** — the real deployment gap to close.
