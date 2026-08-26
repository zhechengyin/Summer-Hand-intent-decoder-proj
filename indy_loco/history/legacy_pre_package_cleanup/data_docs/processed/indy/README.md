# Processed Indy Sessions

Each session is stored as one compressed NPZ file. The 37 sessions are divided chronologically:

| Split | Sessions | Dates | Historical use |
|---|---:|---|---|
| `train/` | 29 | April–October 2016 | weight fitting and training-only statistics |
| `validation/` | 4 | December 2016 | model and checkpoint selection |
| `test/` | 4 | January 2017 | consumed once in Phase 2 |

Important arrays include raw 40 ms spike counts, velocity targets, timestamps, channel metadata, and the stored 32-channel mapping. The active decoder consumed only selected counts plus causal EWMA features; channel names, areas, and unit counts were metadata rather than learned inputs.

See `manifest.json` for the generated session inventory and `../../../README.md` for archive-wide data rules.
