# Resume checklist (2026-07-13)

Stopped 4 concurrent experiments early (CPU overload). **Run only 1–2 at a time.**
Each script re-runs all its configs from scratch (no per-config checkpoint), so to
save time, trim to the configs still needed (noted below). Partial results captured
in DAILY_LOG LOG-059.

## Remaining work, by priority

1. **iter14 architecture** — 3 of 6 models left: `dws_tcn`, `gru_bidir`, `gru_causal`.
   Done: tcngru_bidir 0.677 (non-causal ref), tcngru_causal 0.606, causal_tcn 0.578.
   `py research/iter14_architecture.py` (or trim MODELS to the 3 remaining).
   Then, since causal lost ~0.07: run **bounded-lookahead (model 6)** —
   `A.build_lookahead_tcngru(2)` (80 ms) and `(5)` (200 ms) — the builder is ready.

2. **iter13 follow-up (best lead)** — low-freq (0.2–3 Hz) power selection beat firing
   (0.631 vs 0.577) on 24-session selection. Test **lowfreq selection on the base-6
   sessions** and decode vs firing-6 (0.655). If it wins, update the model-of-record
   channels. (fftweighted config is optional — expected ≈ firing.)

3. **iter11 binning** — 5 of 6 configs left: 20 / 40_ref / 80 / 80-40 / 80-20 ms.
   Done: 10ms_cont = 0.642. The 40ms_ref and the 80/40 & 80/20 overlaps are the
   informative ones (overlapping windows = paper-style decoupling). Fine bins are
   slow (T large) — consider dropping 10ms and running the rest.

4. **iter10 capacity** — effectively DONE. xxwide (400 KB) overfit (0.667). Sweet
   spot is ~xwide (220 KB, 0.679). xxwide_rf optional (likely also overfits).

## Standing conclusions so far (8-ch, 24 sessions)
- Model of record: `models/tcn_gru_8ch/` wide bidir, R²≈0.67, ~100 KB int8. Unchanged.
- 400 KB budget: better spent on ensemble or more data than one big model (single-model
  capacity plateaus ~220 KB then overfits).
- Channel selection: low-freq power is a promising alternative to firing rate (verify on base-6).
- Causal (real-time) costs ~0.07 R²; bounded-lookahead is the likely best real-time compromise.
