# Active decisions

## Data lifecycle

- `data/raw/`: immutable source data.
- `data/processed/`: generated model-ready data.
- `data/processing/`: versioned scripts that convert raw data into processed data.
- Dataset session registries live in `configs/datasets/`, outside the data tree.
- Never convert MAT to CSV merely for training convenience; preserve MAT as the
  raw source and produce NPZ/Parquet/CSV only when a documented consumer needs it.

## Dataset separation

Indy/Loco spike-count reaching experiments and U-M Deep Blue SBP finger-group
experiments remain separate configurations and models. Shared architecture does
not imply interchangeable inputs or labels.

## Causality

Centered temporal smoothing is prohibited in deployable inputs. A unidirectional
network is not sufficient if preprocessing or normalization reads future samples.
Target velocity also follows the causal rule: forward-only filtering plus backward
difference. During the 60-second normalization warm-up no decoder output is valid.

Known non-causal model implementations are not part of the executable research
surface. Their outcomes remain in `docs/history/`. A few recent superseded
experiment scripts are isolated under root `history/` for provenance and must
never be imported by active code. Supported code under `src/`, `models/indy_32ch/`,
`data/processing/`, `experiments/active/` and `experiments/deepblue/` must pass
`tests/test_causality.py`.

## Evaluation

- Select configurations on validation data only.
- A test session that has been inspected repeatedly is burned.
- Month-level claims require month-level outer folds.
- Detector thresholds require nested or separately held-out validation.

## Training sampling

- The Indy 32-channel decoder uses **session-balanced** training-window sampling.
- This decision is frozen from the CPU seed 42/43/44 comparison: session-balanced
  achieved the lowest pooled December validation loss in all three seeds and the
  highest cross-seed mean pooled and session-macro R².
- Window-weighted and month-balanced sampling remain documented comparison
  baselines, not active tuning choices.
- Hyperparameter optimization must keep the sampler fixed so Optuna trials do not
  confound optimizer/model effects with data-exposure effects.
- Phase-1 seed-42 optimization identifies a low-dropout/high-learning-rate region,
  not a frozen point estimate. Trial 32 is the current objective winner, but the
  top five all hit dropout 0.10 and are nearly tied; run a separate boundary
  refinement before multi-seed confirmation.
- Do not resume the completed Phase-1 SQLite study with changed search ranges.
  A different range is a new protocol and must use a new study name/database.
- January test data remains locked until preprocessing, hyperparameters,
  architecture, epoch-selection policy, and seed protocol are all frozen.

## Model status

No historical checkpoint is promoted or retained as a runnable baseline. The
32-channel pipeline has a frozen session-balanced sampler but remains a candidate
until hyperparameter selection, locked-test evaluation, checkpoint/export, and
hardware evidence exist.
