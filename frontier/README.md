# Frontier — active work

**Decoding continuous fingertip velocity from intracortical spikes** (nonhuman
primate M1, O'Doherty/Sabes indy+loco reaching dataset, Zenodo 3854034). This is
the current thrust: a compact, real-time TCN+GRU decoder, now being pushed toward
an **8-channel hardware target**.

Current best: held-out cross-session **r = 0.87** (96 electrodes); **0.76** at
8 electrodes. See [`best_model/`](best_model/) for the exact spec and a saveable
checkpoint.

## Shared core

- **[`core.py`](core.py)** — the architecture + training/eval primitives every
  script imports: `build_net` (TCN+GRU), `run_nn`, `run_linear`, `corr`, `BASE`.
  (Extracted from the original WAY-EEG-GAL script, now in `legacy/`.)

## Scripts

| Script | What it does |
| --- | --- |
| **`crosssession.py`** | The main result: pool 6 indy sessions, evaluate on 2 **held-out** sessions (per-electrode features → true cross-session generalisation). |
| `velocity.py` | Within-session decode (sorted units, 5-block CV). |
| `nch.py` | Electrode-count sweep (8/16/32/96) — the hardware cost curve (LOG-042). |
| `chan_select.py` | **Which 8 channels?** random vs firing-rate vs learned L1-gate selection. |
| `tune.py` | Rate-smoothing σ + causal/real-time sweep (LOG-032). |
| `vellp.py` | Velocity-target low-pass sweep (LOG-030). |
| `subbin.py` | Does within-bin spike *timing* beat counts? (no, for slow velocity — LOG-039). |
| `slow_fast.py` | Slow/fast velocity decomposition test (LOG-034). |
| `activation.py` | Activation-function sweep on the monkey pipeline (→ ReLU, LOG-037/038). |
| `multi.py` | Resumable per-session runner. |

## Run

```bash
py frontier/crosssession.py     # headline held-out metric (~0.87)
py frontier/nch.py              # electrode-count cost curve
py frontier/chan_select.py      # 8-channel selection strategies
```

Data auto-downloads on demand into `data/indy_loco/` (each script has a `fetch`).
Metrics are written to `results/metrics/`.

## Open threads

- **8-channel selection:** does learned/adaptive selection beat firing-rate top-8?
- **Adaptive/switching gate** for non-stationarity (electrode drift) — GRU
  usability score + top-8 switch (bandit / stochastic-gate framing).
- Raw-voltage (Option 2) is **not** testable on this dataset (spikes + waveform
  snippets only, no continuous broadband).
