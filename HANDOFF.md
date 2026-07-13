# HANDOFF — start here (updated 2026-07-13)

Self-contained pickup notes for a fresh session. For the full research trail see
`project_memory/DAILY_LOG.md` (LOG-NNN) and `project_memory/SUMMARY.md`.

## Project in one paragraph

Decode a monkey's (NHP indy/loco) 2D finger **velocity** from intracortical spikes.
Hardware now dictates the target: **8 channels** (spike detection), must run on an
**STM32-class MCU**, budget ~**400 KB**. Metric of record is **R²** (coeff. of
determination on an untouched test session). Only the monkey dataset matters.

## Current model of record — `models/tcn_gru_8ch/`

- Wide bidirectional TCN+GRU, `F64/H64/L1/dils[1,2,4,8]`, ~100k params.
- 8 fixed channels `[26,51,53,66,71,73,75,94]` (top-8 firing on the base 6 sessions).
- Trained on **24 indy sessions**. **TEST R² ≈ 0.67** (0.668 checkpoint / 0.677 harness).
- **int8 = ~100 KB, lossless.** A smaller `F32/H32` variant = ~25 KB int8 at R² 0.655.
- `checkpoint.pt` saved. Reproduce: `py models/tcn_gru_8ch/evaluate.py`.

## What we learned (do / don't)

- ✅ **More training data is the #1 lever**: 6→24 sessions = 0.529→~0.67.
- ✅ **int8 quantization is free** (no R² loss).
- ✅ **Low-freq (0.2–3 Hz) power channel selection beat firing rate** (0.631 vs 0.577,
  24-sess) — TOP OPEN LEAD; verify on the base-6.
- ❌ Bessel **output filter** — redundant (we low-pass the velocity target at 3 Hz).
- ❌ Cheap training tricks (corr loss, aug, reg) — wash.
- ❌ **Re-selecting channels** on more data — overfits (0.628→0.502). Pick once, don't re-select.
- ❌ **One big model to fill 400 KB** — overfits (xxwide 388 KB = 0.667 < xwide 0.679).
  Spend the budget on an **ensemble** or **more data** instead.
- ⚠️ **Causal (real-time) costs ~0.07 R²** (bidir 0.677 → causal 0.606). Bounded-lookahead
  is the likely best real-time compromise (builder ready: `research.architectures.build_lookahead_tcngru`).

## Partial results from 4 experiments stopped early (CPU overload, 2026-07-13)

| exp | done | key numbers | remaining |
| --- | --- | --- | --- |
| iter10 capacity | 3/4 | wide 0.677, xwide(219KB) 0.679, xxwide(388KB) 0.667 | xxwide_rf (skip) |
| iter13 selector | 3/4 | firing 0.577, **lowfreq 0.631**, velcorr 0.582 | fftweighted |
| iter14 arch | 3/6 | bidir 0.677/8.2ms, causal 0.606/5.6ms, causal_tcn 0.578/2.6ms | dws_tcn, gru_bidir, gru_causal, +lookahead |
| iter11 binning | 1/6 | 10ms 0.642 | 20/40/80/80-40/80-20 ms |

## Resume queue

See **`research/RESUME_TOMORROW.md`** for the ordered checklist. Priority: (1) finish
iter14 architecture + run bounded-lookahead; (2) verify low-freq channel selection on
base-6; (3) finish iter11 binning overlaps. **Run only 1–2 experiments at a time — this
box (16 core, ~6 GB free) overloaded at 4 concurrent.**

## Repo map

- `models/tcn_gru_8ch/` — deployment model of record (8ch STM32). `tcn_gru/` = 96ch reference.
- `research/` — active experiments (`harness.py` is the rig; `iter*.py`; `architectures.py`).
- `data/source_data/indy_loco/` — raw .mat (gitignored); `data/processed/` — binning method packages.
- `legacy/` — archived EEG/fNIRS work and old monkey trials.
- `project_memory/` — DAILY_LOG.md + SUMMARY.md (the research record).

## Run notes

- `py research/harness.py` — baseline (96ch + 8ch, reports r and R²).
- Commits are **local** on the current branch; not pushed. Autopilot permissions are broad
  in `.claude/settings.local.json` (bypass mode) — tighten if desired.
