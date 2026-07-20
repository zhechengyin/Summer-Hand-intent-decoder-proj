# Current project status

Last audited: 2026-07-20.

This is the only document intended to state the current project truth. Historical
claims remain in `docs/history/EXPERIMENT_LOG.md` for provenance.

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
- The notebook has produced all 37 processed session artifacts and
  `manifest.json` in the chronological 29/4/4 train/validation/test layout.
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
- The corrected 32-channel causal model has now been trained on the complete
  29-session training split and evaluated on all four December validation
  sessions using CPU. A controlled seed 42/43/44 comparison froze the training
  sampler as **session-balanced**: it won 3/3 seeds, with mean validation R²
  0.5342 +/- 0.0198 and normalized MSE 0.5074 +/- 0.0221. Window- and
  month-balanced sampling are no longer active candidates.
- The four January sessions remained locked and were not loaded during the
  sampling comparison.
- `models/indy_32ch/sweep_phase1_optuna.py` is ready for the first validation-only
  search. It is self-contained, keeps session balancing fixed, tunes learning
  rate/AdamW weight decay/dropout jointly, and never loads January arrays.
- Historical outcomes and caveats are preserved in
  `docs/history/ARCHIVE_RETIREMENT.md` and `docs/history/EXPERIMENT_LOG.md`.

## What does not yet exist

- No promoted 32-channel checkpoint or int8 export.
- The Phase-1 Optuna implementation exists, but no Optuna trial has been run.
  Learning rate, weight decay, dropout, augmentation, and model capacity remain
  at baseline values rather than validated optima.
- `session_inventory.csv`, which the processing notebook is designed to emit,
  is currently absent from the processed Indy directory. The 37 NPZ files and
  manifest used by training are present; regenerate the inventory on the next
  notebook audit rather than reconstructing it manually.
- The complete-pool result is a chronological December validation benchmark,
  not nested month cross-validation and not a January test result.
- No independently validated drift threshold. The historical 0.65 threshold was
  chosen after observing the same 25 sessions.
- No evidence that a fixed 60-second observation is sufficient; the historical
  detector used half of each session.
- Pulled commits do not include their ignored metrics JSON/run logs, so historical
  numerical claims have not been rerun on this checkout.

## Current model designation

| Path | Status | Use |
| --- | --- | --- |
| `models/indy_32ch/` | active candidate; sampler frozen | Session-balanced causal TCN+GRU pending hyperparameter selection, locked-test evaluation, export, and hardware validation. |

No historical checkpoint is a runnable model of record.

## Supported executable surface

- `src/intent_decoder/`: reusable causal implementation.
- `data/processing/indy_loco/indy/`: reproducible Indy audit and conversion notebook.
- `experiments/active/`: decision-changing Indy evaluation only.
- `experiments/deepblue/`: separate-input U-M benchmark.
- `models/indy_32ch/sweep_phase1_optuna.py`: self-contained active model-selection
  entry point.

Superseded recent training scripts are isolated under root `history/` and are
not imported by active code. Older archive/legacy/compatibility code remains
deleted after documentation.

## Current research question

Can session-balanced training improve the corrected 32-channel causal decoder
through validation-only hyperparameter optimization while retaining robust
performance on every December session? January remains locked until the full
configuration is frozen.
