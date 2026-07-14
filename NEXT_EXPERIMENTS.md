# NEXT_EXPERIMENTS — scratch resume doc (TEMPORARY, delete when multiscale lands)

> Throwaway pickup notes for a **fresh chat with clean context**. Durable record is
> `project_memory/DAILY_LOG.md` (LOG-NNN) + `SUMMARY.md` + `HANDOFF.md`. Delete this
> file once the multiscale work is adopted into the model of record.

## 30-second catch-up

- Task: decode monkey 2D finger **velocity** from **8 spike-detection channels**,
  **STM32**, **real-time** ⇒ decoder must be **strictly causal** (no future).
  Metric = **R²** on untouched `test1`. Only the indy/loco monkey data matters.
- Deployable model of record: `models/tcn_gru_8ch/` — causal wide TCN+GRU
  (`bidir=False`), 8 firing-selected channels `[26,51,53,66,71,73,75,94]`,
  24 sessions, 40 ms bins. **Causal R² ≈ 0.606** (bidirectional 0.677 is an OFFLINE
  reference only — not deployable).
- **The live lever (yesterday's breakthrough, LOG-068/069):** feeding each channel
  at **multiple causal timescales** (raw + slow EWMA) instead of one 40 ms rate.
  3-seed causal: single-scale 0.618 → **multiscale 0.646 (+0.028, confirmed real)**.
  Benefit **saturates at 2 scales** (raw + one EWMA α≈0.2, 16 features, ~75 KB int8).

## RESUME HERE (the multiscale thread — highest priority)

1. **(the one we stopped mid-run)** `research/iter22_scale_sweep.py` was running the
   6-scale (`ms6`) config when we stopped for a clean break — only `ms2` (=0.646)
   finished. Optionally rerun `ms6` to confirm 6 scales don't beat 2/4 (expected: no).
2. **Tune the single EWMA α** for the 2-scale config (raw + one EWMA). α=0.2 gave
   0.646; sweep ~{0.1, 0.15, 0.2, 0.3} at 3 seeds to find the best slow timescale.
   Reuse `research/iter20_multiscale.py` `ewma_feats()` + `research/harness.py`.
3. **ADOPT 2-scale multiscale into `models/tcn_gru_8ch`** (the deliverable): add the
   EWMA feature step to `config.py`/`evaluate.py` (input channels 8→16), run
   `train_and_save.py` (writes the **causal + multiscale** checkpoint — note the
   current `checkpoint.pt` is still the OLD bidirectional one), then `export_int8.py`.
4. Decide **single model (~0.61)** vs **3-seed ensemble (0.646)** (3× size, still
   STM32-OK) as the shipped config.

## Other untried directions (after multiscale, roughly ranked)

- **LFADS-lite / latent dynamics** — auxiliary head that reconstructs smoothed rates
  so the GRU latent captures population dynamics. Biggest theoretical upside for
  reaching tasks; heaviest build; make it causal + small.
- **Multi-task** — predict velocity + position (integral) and/or speed jointly as
  auxiliary targets to regularize. Cheap-ish (second head + combined loss).
- **Proper Kalman / state-space post-filter** (learned, causal) — we only tried a
  plain forward-EMA (hurt); a real state-space with a velocity dynamics model is
  untested.
- **SNN** (spiking) — the paper's low-power option; untested here.
- **Data/hardware levers (out of model scope, but the real ceiling):** more channels,
  sessions closer in time to `test1`, per-user calibration on top of the 24-session
  pool, or richer signal (broadband — not in this dataset).

## DEAD ENDS — do NOT re-run (already ruled out; see DAILY_LOG)

- Architecture: LSTM (ties GRU), Transformer (worse), plain-CNN+GRU (worse),
  TCN-only / GRU-only (worse). TCN+GRU wins. (LOG-060/066)
- Capacity: plateaus ~220 KB, overfits at 400 KB. (LOG-061)
- More data: 24 nearby sessions is the limit; distant sessions hurt. (LOG-065)
- Channel selection: firing-8 on base-6 is best; learned/low-freq/fft/re-selection
  all lose or tie. (LOG-063)
- Binning: 40 ms boxcar best; finer/coarser worse; overlapping windows don't help. (LOG-061)
- Output smoothing (Bessel / EMA): redundant/hurts (target already 3 Hz low-passed). (LOG-050/064)
- Causal arch tweaks (more RF, depth, width): no gain. (LOG-064)
- Cheap training tricks (correlation loss, stronger aug, more reg): wash. (LOG-051)
- Bidirectional / bounded-lookahead: NOT deployable (40 ms/bin ⇒ latency too high). (LOG-062)

## How to run

- Rig: `research/harness.py` — `prep_more(...)` / `H.run(data, cfg, seeds=(...), build=...)`
  returns r & R². `models/tcn_gru/best_model.py` = `build_net` + `r2`/`corr`.
  `models/tcn_gru/evaluate.py` = `load_electrode`, `E.TRAIN/EVAL/TEST`. 24-session pool
  = `E.TRAIN + research/iter7_final.py::EXTRA18`.
- **Memory:** run only **1–2 experiments at a time** — box OOMs ~2 GB free during
  24-session loads; stagger launches.
- **Logging (standing directive):** every result → append a DAILY_LOG entry
  (LOG-NNN) **with a `Files:` section** (files used + one-line purpose each) → update
  SUMMARY if best changes → `git commit` (+push). Commit msgs end with the
  Co-Authored-By line.
