# Current Status

Last audited: 2026-07-24.

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

Phase 3a now provides a label-free 60-second compatibility gate:

- layer A compares log-rate shape, robust rate distance, global rate ratio and
  unexpectedly silent channels against multiple historical month references;
- layer B projects all 1,500 prefix bins into a fixed five-dimensional PCA space
  and compares full mean/covariance distributions with Gaussian KLD;
- global and multi-month KLD are both reported instead of assuming that
  multi-reference is better;
- one abnormal evidence family produces `warning`; at least two produce
  `abstain`; KLD alone cannot abstain;
- the loader accepts only the 29 train and four December validation sessions,
  and the model code hard-fails on January-or-later session names.

Thresholds use a conservative 0.99 empirical quantile and are recalibrated by an
inner leave-one-complete-month-out loop inside every outer held-month fold.
January and velocity labels are never loaded.

Initial pre-January results are developmental, not a frozen detector:

- intact held-month sessions: combined flag 5/33 (15.2%), abstain 3/33 (9.1%);
- global KLD and multi-reference KLD each flagged 1/33 intact sessions;
- 25% synthetic channel dropout: combined flag 33/33;
- mixed 65% thinning plus 25% channel dropout: combined flag 33/33 and abstain
  24/33;
- multi-reference KLD did not outperform global KLD in these stress tests.

The intact-data abstain rate is still too high to freeze without performance
labels. Phase 3b must obtain strict held-month decoder R² by retraining a
temporary decoder without each month, then determine whether the three flagged
intact sessions were genuinely decoder-incompatible. Do not load January for
this work.

## Supported files

- `data/processing/indy_loco/indy/causal_targets.py`
- `data/processing/indy_loco/indy/prepare_indy_model_ready.ipynb`
- `models/indy_32ch/input_pipeline.py`
- `models/indy_32ch/features.py`
- `models/indy_32ch/model.py`
- `models/indy_32ch/drift_detector.py`
- `models/indy_32ch/sampling.py`
- `experiments/active/phase0a_data_audit.py`
- `experiments/active/phase0a_data_audit.ipynb`
- `experiments/active/phase3a_drift_detector.py`
- `experiments/active/test_phase3a_drift_detector.py`
- `models/indy_32ch/checkpoint.pt`

Everything under `history/` is provenance only and must not be imported.
