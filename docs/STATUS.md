# Current Status

Last audited: 2026-07-23.

## Objective

Decode two-axis fingertip velocity from intracortical spike counts on an
STM32-class target. The complete pipeline must remain causal.

## Data

- Dataset: Indy from Zenodo record 3854034.
- Source files: 37/37 under `data/raw/indy_loco/indy/`; raw data is immutable.
- Processed schema: `indy_counts_velocity_v2`.
- Split: 29 train sessions (April--October 2016), 4 validation sessions
  (December 2016), and 4 consumed locked-test sessions (January 2017).
- Model input contains only 96-channel count arrays before fixed-32 selection.
- Target is two-axis causally filtered backward-difference velocity.
- The existing audit found no checksum, schema, shape or non-finite-value
  failures. Neural channel statistics nevertheless drift materially by month.

## Frozen model

| Item | Value |
| --- | --- |
| Checkpoint | `models/indy_32ch/checkpoint.pt` |
| SHA-256 | `2ee52c426ee43ba88cebe7c85dd8392f40f9e75748abe9bbf4e94093556363a5` |
| Parameters | 78,786 |
| Input | 32 counts + 32 causal-EWMA features |
| Window | 50 bins / 2 seconds |
| Warm-up | first 60 seconds, outputs discarded |
| Model | causal TCN + unidirectional GRU |
| Sampling | session-balanced |
| Seed | 43 |
| Checkpoint epoch | 7 |
| Learning rate | 0.0009 |
| Weight decay | 0.060 |
| Dropout | 0.025 |

The exact channels and remaining settings are in `configs/indy_32ch.yaml`.

## Final evidence

- Session-balanced sampling won all three controlled seed comparisons and is
  frozen.
- Five-seed regularization confirmation selected weight decay 0.060 by the
  preregistered rule.
- Frozen December validation: loss 0.480166, pooled R² 0.560362.
- Locked January: loss 0.422562, pooled R² 0.551146, session-macro R² 0.504789.
- January session R² values:
  - `indy_20170123_02`: 0.694701
  - `indy_20170124_01`: -0.052402
  - `indy_20170127_03`: 0.698878
  - `indy_20170131_02`: 0.677979

The model generalizes well on compatible sessions but has a severe
session-level tail. January is consumed and cannot be used again for model,
feature or detector selection.

## Active work

Build a new label-free drift detector from scratch. It must:

1. use only the 33 pre-January sessions;
2. use the first 60 seconds of fixed-32 counts;
3. evaluate firmware-feasible rate correlation, robust distance, silent-channel
   count and rate-ratio features;
4. select features and thresholds inside nested leave-one-month-out validation;
5. prioritize avoiding false negatives and report sensitivity, specificity,
   balanced accuracy, accepted-session coverage/R² and firmware cost.

No active detector script exists yet. The previous detector scaffold was deleted
because it used an obsolete pool and could reintroduce January into training.

## Supported files

- `src/intent_decoder/data/indy.py`
- `src/intent_decoder/features/causal.py`
- `src/intent_decoder/model/tcn_gru.py`
- `src/intent_decoder/sampling.py`
- `data/processing/indy_loco/indy/prepare_indy_model_ready.ipynb`
- `experiments/active/indy_month_drift_analysis.py`
- `experiments/active/indy_month_drift_analysis.ipynb`
- `models/indy_32ch/checkpoint.pt`

Everything under `history/` is provenance only and must not be imported.
