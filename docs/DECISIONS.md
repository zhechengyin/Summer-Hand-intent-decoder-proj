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

Known non-causal archive/model implementations are not retained as executable
code. Their outcomes remain in `docs/history/`. Supported code under `src/`,
`data/processing/`, `experiments/active/` and `experiments/deepblue/` must pass
`tests/test_causality.py`.

## Evaluation

- Select configurations on validation data only.
- A test session that has been inspected repeatedly is burned.
- Month-level claims require month-level outer folds.
- Detector thresholds require nested or separately held-out validation.

## Model status

No historical checkpoint is promoted or retained as a runnable baseline. The
32-channel pipeline remains a candidate until checkpoint/export/hardware evidence exists.
