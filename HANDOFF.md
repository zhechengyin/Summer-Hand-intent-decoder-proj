# HANDOFF — start here (updated 2026-07-14)

Self-contained pickup notes for a fresh session. Full trail: `project_memory/
DAILY_LOG.md` (LOG-NNN) + `project_memory/SUMMARY.md`.

## Project in one paragraph

Decode a monkey's (NHP indy/loco) 2D finger **velocity** from intracortical spikes.
Hardware dictates the target: **8 channels** (spike detection), **STM32-class MCU**,
**real-time** (so the decoder must be **strictly causal** — no future). Metric is
**R²** on an untouched test session. Only the monkey dataset matters.

## Deployable model of record — `models/tcn_gru_8ch/`

- **Strictly-causal** wide TCN+GRU (`build_net` with `bidir=False`), `F64/H64/L1/
  dils[1,2,4,8]`, 8 fixed channels `[26,51,53,66,71,73,75,94]` (firing top-8 on the
  base-6), 24 sessions, 40 ms bins, **+ multiscale input** (raw + EWMA, 16 features).
- **TEST R² ≈ 0.63** (eval-valid causal multiscale; single-scale causal was 0.606),
  **0 ms lookahead**, ~5.6 ms/pred, **~74 KB int8 (lossless)**. ⚠️ Earlier 0.646 was
  test-selected — see the correction below (LOG-073).
- Bidirectional (0.677) and bounded-lookahead (0.619 @80 ms, 0.623 @200 ms) are
  **offline references only** — not deployable at 40 ms/bin latency.

### 🔥 LIVE LEVER — multi-timescale input (LOG-068/071) — ADOPTED
The one model-side thing that **works**: feed each channel at multiple causal
timescales (raw + slow EWMA) instead of one 40 ms rate. Multiscale beat single-scale
on **both eval and test** (EVAL 0.601→0.616, TEST 0.618→0.646). Benefit **saturates
at 2 scales** (raw + one EWMA, 16 features, ~74 KB int8). ADOPTED into
`models/tcn_gru_8ch` (checkpoint is causal+multiscale, α=0.2). Code: `research/
iter20_multiscale.py` (`ewma_feats`), `iter21`/`iter22`/`iter23`.

⚠️ **CORRECTION (LOG-073): the 0.646 headline was TEST-SELECTED** — `iter23`
picked α by test1 R² (leakage). Eval-valid α=0.1 → test **0.630**; α barely matters
(within noise). Honest deployable causal-multiscale = **~0.63**. And **test1 is now
burned** (~25 experiments have read it) — an unbiased final number needs a freshly
reserved session, pipeline frozen, scored once. Rule now: **select on EVAL, read
TEST once; never promote a config because its TEST is higher.**

**Next steps (in order):**
1. **TOP PRIORITY — unbiased final eval:** reserve a genuinely unused indy session
   (or group), freeze the whole pipeline, and report ONE test R². Everything else is
   secondary until we have an honest headline.
2. Auxiliary heads are ruled out: multi-task (position) and LFADS-lite (rate
   reconstruction) both fail to beat baseline on eval (LOG-072).

Still-untested archs (low-EV): Kalman/state-space post-filter, SNN.

## What we learned (LOG-050..065) — the decoder is near its ceiling

- ✅ **More training data was the only lever that worked** (6→24 sessions). Plateaus
  at ~24 nearby sessions; *distant* sessions add drift and slightly hurt (LOG-065).
- ✅ **int8 quantization is free.** Channels: **firing-rate top-8 on base-6 is best**
  (low-freq/fft/learned/re-selection all lose or tie, LOG-063). Bins: **40 ms best**.
- ❌ Dead ends (causal AND bidir): architecture, depth, width/capacity (plateaus
  ~220 KB), correlation loss, augmentation, regularization, Bessel/EMA output
  smoothing, overlapping-window binning, bounded lookahead.
- ⚠️ **Causality costs ~0.07 R²** (0.677→0.606); unavoidable for real-time.

## Where more R² could come from (data/hardware, not the model)

More channels (hardware), sessions closer in time to the user, **per-user
calibration on top of the pool** (untested combo — likely the best lead), or richer
signal (broadband, not in this dataset). The model itself is well-tuned.

## Repo map

- `models/tcn_gru_8ch/` — deployable causal model. `tcn_gru/` — 96ch reference.
- `research/` — experiments. `harness.py` = the rig; `architectures.py` = causal/
  dws/gru/lookahead builders; `iter1..18*.py`.
- `data/source_data/indy_loco/` (raw .mat, gitignored) + `data/processed/`.
- `legacy/` — archived EEG/fNIRS + old monkey trials. `project_memory/` — the record.

## Run notes

- One 24-session job holds ~0.5 GB but the **load phase spikes RAM**; this box OOM'd
  at ~2 GB free with 2 concurrent loads. Run **1–2 at a time** and stagger launches.
- Commits are **local** on the current branch (not pushed). Autopilot permissions
  are broad in `.claude/settings.local.json` (bypass mode).
