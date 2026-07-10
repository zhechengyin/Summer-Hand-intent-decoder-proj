# Neural Intent Decoder — hand/finger velocity from neural activity

Decoding **continuous hand/finger velocity** from neural signals with a compact,
real-time-oriented **TCN+GRU** sequence decoder:

> Given neural activity over a short window, estimate the current fingertip
> velocity vector.

## Active frontier → [`frontier/`](frontier/)

**Intracortical spike → fingertip velocity** (nonhuman-primate M1, indy+loco
reaching data). This is the current work, now targeting an **8-channel hardware
front-end**.

| Setting | Held-out cross-session r |
| --- | ---: |
| Full 96 electrodes | **0.87** |
| 8 electrodes (top-8 by firing rate) | 0.76 |

The current best model — architecture, exact config, metrics, and a saveable
checkpoint — lives in **[`frontier/best_model/`](frontier/best_model/)**.

```bash
py frontier/crosssession.py          # reproduce the 0.87 held-out result
py frontier/nch.py                   # electrode-count cost curve (8→96)
py frontier/chan_select.py           # which 8 channels? (firing-rate wins)
py frontier/best_model/train_and_save.py   # -> frontier/best_model/checkpoint.pt
```

## Repository map

```
frontier/          ACTIVE — monkey intracortical decoding (the 8-channel work)
  core.py            shared architecture: build_net (TCN+GRU), run_nn, corr, BASE
  crosssession.py    headline held-out cross-session result (0.87)
  nch.py             electrode-count sweep       chan_select.py  channel selection
  velocity.py tune.py vellp.py subbin.py slow_fast.py activation.py multi.py
  best_model/        current best: config.py, README, train_and_save.py, checkpoint.pt

legacy/            CONCLUDED — original EEG+fNIRS intent decoder (near-chance MI)
  src/, main.py, config.yaml, tools/   (self-contained; see legacy/README.md)

project_memory/    research record — SUMMARY.md (current state) + DAILY_LOG.md (LOG-NNN)
data/              datasets (gitignored; auto-downloaded on demand)
results/           metrics + figures (gitignored)
```

Findings and rationale are logged in
[`project_memory/DAILY_LOG.md`](project_memory/DAILY_LOG.md); the current state is
summarised in [`project_memory/SUMMARY.md`](project_memory/SUMMARY.md).

## Notes

- The decoder generalises **across sessions** of the same subject but **not
  across subjects** (indy→loco collapses) — per-subject calibration is required.
- This dataset has spike times + waveform snippets only (no continuous broadband
  voltage), so raw-voltage-in decoding cannot be benchmarked here.
- `requirements.txt` covers both frontier and legacy.
