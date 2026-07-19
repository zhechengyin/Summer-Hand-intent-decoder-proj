# Current project status

Last audited: 2026-07-18.

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
- All 37 official Indy MAT sessions are present under the immutable canonical path
  `data/raw/indy_loco/indy/`. The versioned dataset registry contains the matching
  Zenodo MD5 for every session.
- `data/processing/indy_loco/indy/prepare_indy_model_ready.ipynb` is the supported
  raw-data preprocessing surface. It validates the source map and stores only a
  stable 96-channel M1 count input plus two-axis velocity target in each NPZ under
  `data/processed/indy_loco/indy/{train,validation,test}/` using a fixed
  chronological 29/4/4 session split. January 2017 is the locked test month.
- The loader now samples kinematics using the latest already-observed sample at each
  bin end. It no longer uses linear interpolation across a future 250 Hz sample.
- An eight-session causal smoke test now trains the 32-channel candidate end to
  end. On the fixed 6/1/1 diagnostic split, eval selected epoch 32 with eval R²
  0.5781 and reused-test1 R² 0.5851. This confirms the pipeline trains, but is
  not an unbiased generalization result.
- The retired 8-channel checkpoint record had 75,714 parameters and reported
  test R² 0.6325, but its centered-Gaussian input preprocessing used future samples.
- Historical research supports counts + causal EWMA and 32 channels as the most
  promising candidate; its code is retired and its old scores are not reusable.
- Historical outcomes and caveats are preserved in
  `history/ARCHIVE_RETIREMENT.md` and `history/EXPERIMENT_LOG.md`.

## What does not yet exist

- No promoted 32-channel checkpoint or int8 export.
- The new 37-session preprocessing notebook has intentionally not been executed;
  the user will generate and validate the new processed artifacts locally.
- The corrected causal implementation has not yet been benchmarked across the
  complete session pool. The eight-session smoke test cannot replace month-CV,
  and old scores cannot be reused because the target-velocity definition and
  normalization protocol changed.
- No independently validated drift threshold. The historical 0.65 threshold was
  chosen after observing the same 25 sessions.
- No evidence that a fixed 60-second observation is sufficient; the historical
  detector used half of each session.
- Pulled commits do not include their ignored metrics JSON/run logs, so historical
  numerical claims have not been rerun on this checkout.

## Current model designation

| Path | Status | Use |
| --- | --- | --- |
| `models/indy_32ch/` | candidate slot | Destination for the future validated causal checkpoint and manifest. |

No historical checkpoint is a runnable model of record.

## Supported executable surface

- `src/intent_decoder/`: reusable causal implementation.
- `data/processing/indy_loco/indy/`: reproducible Indy audit and conversion notebook.
- `experiments/active/`: decision-changing Indy evaluation only.
- `experiments/deepblue/`: separate-input U-M benchmark.

The former archive/legacy/compatibility code was deleted after documentation.

## Current research question

Can the corrected 32-channel end-to-end causal pipeline retain useful month-level
performance and support a drift threshold selected without outer-fold leakage?
