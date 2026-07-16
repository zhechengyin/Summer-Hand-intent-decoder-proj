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
- The supported pipeline is causal by construction: unsmoothed counts, causal
  EWMA, forward-only target filtering, backward-difference velocity, a frozen
  60-second past-only normalization prefix, causal TCN padding and a
  unidirectional GRU.
- `tests/test_causality.py` verifies prefix invariance and rejects known
  future-data operations in supported code.
- The canonical data path is `data/raw/indy_loco/`; the original eight aliased MAT
  sessions have been moved there locally.
- Those eight sessions have been regenerated locally under
  `data/processed/indy_loco/bin_40ms_causal_counts/` with causal target metadata.
- The historical 8-channel checkpoint has 75,714 parameters and records test R²
  0.6325, but its centered-Gaussian input preprocessing used future samples.
- Archived research supports counts + causal EWMA as the preferred input feature.
- Archived results suggest 32 channels are substantially more robust than 8 and
  that a prediction-variance proxy may identify drifted sessions.

## What does not yet exist

- No promoted 32-channel checkpoint or int8 export.
- The corrected causal implementation has not yet been benchmarked across the
  complete session pool. Old scores cannot be reused because the target-velocity
  definition and normalization protocol changed.
- No independently validated drift threshold. The historical 0.65 threshold was
  chosen after observing the same 25 sessions.
- No evidence that a fixed 60-second observation is sufficient; the historical
  detector used half of each session.
- Pulled commits do not include their ignored metrics JSON/run logs or most raw
  sessions, so numerical claims have not been rerun on this checkout.

## Current model designation

| Path | Status | Use |
| --- | --- | --- |
| `models/tcn_gru/` | historical reference | Old 96-channel code; may reproduce non-causal protocols and is not imported by supported code. |
| `models/tcn_gru_8ch/` | legacy checkpoint | Reproduction baseline only; not fully causal end to end. |
| `models/indy_32ch/` | candidate slot | Destination for the future validated causal checkpoint and manifest. |

## Current research question

Can the corrected 32-channel end-to-end causal pipeline retain useful month-level
performance and support a drift threshold selected without outer-fold leakage?
