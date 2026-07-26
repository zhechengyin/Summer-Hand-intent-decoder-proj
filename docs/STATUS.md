# Current Status

Last audited: 2026-07-26.

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

## Active detector

Phase 3b completed all five strict pre-January leave-one-month-out folds. Every
temporary decoder used seed 43, stopped at fixed epoch 7, excluded the complete
held month from training, and loaded held velocity only after optimization:

- 33-session macro R²: 0.559710;
- pooled R²: 0.526599;
- best held month: September, macro R² 0.735912;
- `indy_20160630_01`: R² -0.136478;
- `indy_20161013_03`: R² -0.054134.

The raw-count Phase-3a gate detected the October failure but missed the June
failure. Its thresholds should not be lowered to fit the known outcomes.

Phase 3c therefore adds a second, decoder-derived layer. During the same gated
60-second prefix the frozen network runs inference but does not release output.
The detector compares:

- five-dimensional PCA summaries of all GRU hidden states;
- the two-dimensional predicted-output distribution;
- predicted-output first differences and 10-second hidden chunks as
  diagnostic-only scores.

All projections, month references and thresholds are fitted from reference
months only. No velocity label enters the detector and no decoder weight is
updated. The second layer vetoes only when hidden-state KLD and absolute-output
KLD both exceed their conservative 0.99 severe thresholds.

Development evaluation:

- both known negative-R² sessions: `abstain`;
- remaining 31 held sessions: `pass`;
- sensitivity: the same separation held for hidden dimensions 3/5/8 crossed
  with covariance shrinkage 0.05/0.10/0.20;
- output-delta and temporal scores produced five extra diagnostic flags, so
  they cannot veto.

The detector and model are now connected by
`models/indy_32ch/runtime.py`. The runtime collects exactly 1,500 count bins,
runs both gate layers, and releases only post-warm-up model predictions when
the combined decision permits it. It never loads velocity and never updates a
weight.

The final references contain 31 compatible development sessions; the two known
negative-R² sessions are excluded from the compatible reference fit. Artifact
reload and an end-to-end December smoke test passed. However, an important
distinction remains:

- strict out-of-month fold decoders: both June 30 and October 13 abstain;
- final integrated active checkpoint: October 13 abstains, June 30 passes.

The active checkpoint had already learned from both sessions, whereas each
outer-fold checkpoint had never seen its held month. Therefore Phase 3 is
complete as a development experiment and integrated as a runtime candidate,
but it is not a demonstrated universal failure detector. January remains
forbidden, and final detector claims require prospective sessions. STM32
memory, timing and fixed-point equivalence are also pending.

## Supported files

- `data/processing/indy_loco/indy/causal_targets.py`
- `data/processing/indy_loco/indy/prepare_indy_model_ready.ipynb`
- `models/indy_32ch/input_pipeline.py`
- `models/indy_32ch/features.py`
- `models/indy_32ch/model.py`
- `models/indy_32ch/drift_detector.py`
- `models/indy_32ch/decoder_state_detector.py`
- `models/indy_32ch/runtime.py`
- `models/indy_32ch/sampling.py`
- `experiments/active/phase0a_data_audit.py`
- `experiments/active/phase0a_data_audit.ipynb`
- `models/indy_32ch/checkpoint.pt`

Everything under `history/` is provenance only and must not be imported.
Completed detector runners and regression checks are in `history/phase3/`.
