# Current project status

Last audited: 2026-07-15.

This is the only document intended to state the current project truth. Historical
claims remain in `history/EXPERIMENT_LOG.md` for provenance.

## Objective

Decode two-dimensional fingertip/reaching velocity from intracortical spike
events on an STM32-class target. The deployable pipeline must be causal from raw
input through normalization and output.

UM Deep Blue is a separate two-finger-group SBP benchmark. It is not combined
with Indy/Loco training because its input feature and behavioral target differ.

## What exists

- Reusable code now lives under `src/intent_decoder/`.
- The canonical data path is `data/raw/indy_loco/`; the original eight aliased MAT
  sessions have been moved there locally.
- The historical 8-channel checkpoint has 75,714 parameters and records test R²
  0.6325, but its centered-Gaussian input preprocessing used future samples.
- Archived research supports counts + causal EWMA as the correct causal feature.
- Archived results suggest 32 channels are substantially more robust than 8 and
  that a prediction-variance proxy may identify drifted sessions.

## What does not yet exist

- No promoted 32-channel checkpoint or int8 export.
- No end-to-end streaming normalization benchmark. Older cross-session scripts
  normalized held-out sessions using whole-session statistics.
- No independently validated drift threshold. The historical 0.65 threshold was
  chosen after observing the same 25 sessions.
- No evidence that a fixed 60-second observation is sufficient; the historical
  detector used half of each session.
- Pulled commits do not include their ignored metrics JSON/run logs or most raw
  sessions, so numerical claims have not been rerun on this checkout.

## Current model designation

| Path | Status | Use |
| --- | --- | --- |
| `models/tcn_gru/` | architecture/reference | Shared readable TCN+GRU implementation and old 96-channel reference. |
| `models/tcn_gru_8ch/` | legacy checkpoint | Reproduction baseline only; not fully causal end to end. |
| `models/indy_32ch/` | candidate slot | Destination for the future validated causal checkpoint and manifest. |

## Current research question

Can a 32-channel counts + causal-EWMA model, normalized only from an initial
60-second observation prefix, retain the reported month-level performance and
support a drift threshold selected without outer-fold leakage?
