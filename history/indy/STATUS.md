# Current Status

Last audited: 2026-07-28.

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

## Retained checkpoints

| Item | Value |
| --- | --- |
| Integrated baseline | `models/indy_32ch/64x64checkpoint.pt` |
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

The completed Phase-4 candidate is
`models/indy_32ch/48x48checkpoint.pt`:

| Item | 48/48 firmware candidate |
| --- | --- |
| SHA-256 | `5c8b375787ff93f90006df5f0cfea07303660928c7b69a84d4d75e1a368319ef` |
| Size | 199,733 bytes |
| Parameters | 45,266 |
| Seed / training budget | 43 / 20 epochs |
| Selected checkpoint epoch | 10 |
| Selection rule | Minimum pooled December validation loss |
| Train pooled R² | 0.763162 |
| December pooled R² | 0.565134 |
| December session-macro R² | 0.575004 |
| December worst-session R² | 0.346125 |

Only the 29 training sessions updated its weights. December was inference-only
and selected the checkpoint epoch; January was never loaded. The 48/48 file is
the preferred standalone firmware candidate because it is 42.5% smaller and
its direct December pooled R² is 0.004772 higher than the 64/64 checkpoint.
The integrated runtime still defaults to 64/64 because the saved Layer-2 drift
reference was fitted to 64-dimensional GRU states.

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

## Completed Phase 4a architecture sweep

Phase 4a completed on 2026-07-27 and its runner is archived at
`history/phase4/phase4a_architecture_sweep.py`. The architecture-only
Optuna study created 30 trial records: 20 complete and 10 pruned, with no
failed or unfinished trials. The protected baseline checksum remained
unchanged and no candidate checkpoint was written.

Exactly five fields vary:

- TCN filters: 32, 48, 64 or 96;
- GRU hidden width: 32, 48, 64 or 96;
- power-of-two TCN dilation blocks: 2, 3 or 4;
- temporal kernel size: 2, 3 or 4;
- GRU layers: 1 or 2.

Learning rate 0.0009, weight decay 0.060, dropout 0.025, seed 43,
session-balanced sampling, 32 channels, causal features, 60-second
normalization, 50-bin windows, seven trained epochs on the frozen 20-epoch
cosine trajectory, and all data rules remain fixed. Each trial uses five
complete pre-January held-month folds. January is rejected before any array is
opened.

Selection score is 75% held-session macro R² plus 25% held-session 10th
percentile R². Worst-session R² and parameter count are retained as guardrails.
Pruning can happen only after complete folds, never inside an epoch. No Phase
4a model weight was saved.

The highest-scoring unique candidate was 48 TCN filters, 48 GRU hidden units,
four TCN blocks, kernel size 3 and one GRU layer. It retains the baseline's
31-bin receptive field and temporal structure while reducing parameters from
78,786 to 45,266.

| Metric | Frozen 64/64 baseline | 48/48 candidate | Candidate minus baseline |
| --- | ---: | ---: | ---: |
| Selection score | 0.499329 | 0.503833 | +0.004505 |
| Session-macro R² | 0.559710 | 0.561457 | +0.001747 |
| Session-q10 R² | 0.318187 | 0.330963 | +0.012777 |
| Worst-session R² | -0.136478 | -0.200013 | -0.063535 |
| Parameters | 78,786 | 45,266 | -42.5% |

The candidate won 16 of 33 paired held sessions and lost 17; its median
session delta was -0.000657. April improved by 0.0171 macro R², but June
declined by 0.0215 and the known June 30 failure worsened. The result therefore
does not establish an accuracy improvement. It nominates a much smaller
non-inferiority candidate.

The search covered only 18 unique architectures out of 288 possible
combinations. Nine completed and nine were pruned. The deterministic 48/48
architecture was repeated in 12 trial records, which are duplicates rather
than independent confirmations. The sweep's strongest structural indication
is to retain four blocks, kernel size 3 and one GRU layer; larger/deeper
variants did not justify their cost.

## Completed Phase 4b five-seed confirmation

Phase 4b completed on 2026-07-28. All 50 planned fits finished: two
architectures, seeds 42--46 and five complete pre-January held-month folds.
January was never loaded, held labels never updated weights or selected an
epoch, and the protected checkpoint checksum remained unchanged.

| Five-seed metric | 64/64 baseline | 48/48 candidate | Candidate minus baseline |
| --- | ---: | ---: | ---: |
| Selection score | 0.477283 | 0.473767 | -0.003516 |
| Session-macro R² | 0.549975 | 0.544056 | -0.005919 |
| Session-q10 R² | 0.259207 | 0.262901 | +0.003694 |
| Worst-session R² | -0.159978 | -0.174609 | -0.014631 |
| Parameters | 78,786 | 45,266 | -42.5% |
| Estimated multiplies / 50 bins | 3,897,600 | 2,232,000 | -42.7% |

All four predeclared non-inferiority checks passed:

- macro R² delta -0.005919, limit -0.010;
- q10 R² delta +0.003694, limit -0.020;
- worst-session R² delta -0.014631, limit -0.020;
- worst-month macro R² delta -0.018616, limit -0.020.

The result nominates 48/48 for firmware efficiency; it does not establish
higher accuracy. The candidate lost a mean 0.0059 macro R², had greater
seed-to-seed variation, and was weakest relative to baseline in June.

Phase 4b itself intentionally wrote no weight. The subsequent fixed builder
completed on 2026-07-28 and produced `models/indy_32ch/48x48checkpoint.pt`.
It used the confirmed architecture, seed 43, the full 20-epoch cosine schedule
and minimum December validation loss. Its epoch-7 CPU metrics reproduced the
corresponding Phase-4b cell exactly, providing an additional protocol check.
The build script and Phase-4b runner are now archived under `history/phase4/`.

## Registered Phase 5a 64-channel comparison

Phase 5a is implemented but has not yet been run. It changes only two controlled
dimensions relative to the frozen training protocol:

- neural input expands from 32 to 64 channels selected from train-session
  60-second prefixes only;
- model width compares TCN/GRU 64/64 against 48/48.

Each selected channel contributes raw counts and one causal EWMA, so the model
receives 128 features per 50-bin window. Both architectures use CPU, seed 43,
session-balanced sampling, learning rate 0.0009, weight decay 0.060, dropout
0.025, four causal dilation blocks and a complete 30-epoch cosine schedule.
They see identical sampled windows in every corresponding epoch.

Only the 29 train sessions fit channels, normalization, targets and weights.
December is inference-only and selects each minimum-loss checkpoint. January
arrays are never loaded. Both experiment checkpoints are written below
`results/indy/phase5a_64channel_width_comparison/checkpoints/`; neither active
32-channel checkpoint is modified. The existing detector is incompatible with
the new 64-channel mapping and must be refitted if a Phase-5a model is promoted.

The initial 2026-07-28 run is invalid because the earlier `auto` device default
selected Apple MPS. Exact epoch-1 reproduction found mean/max pre-clipping
gradient norms of 4,784/6,430 on MPS versus 0.961/1.848 on CPU. MPS Train and
December R² were -0.273/-0.210, while the identical CPU run reached
0.710/0.615. Forward predictions matched to approximately 3e-7; the error is in
MPS backward propagation to the spatial projection. Phase 5a is now CPU-only,
and the invalid result artifacts must be overwritten by the clean CPU run.

## Next gate

Run Phase 5a and compare pooled, session-macro and worst-session December R²
alongside parameter count. Do not promote the single-seed winner immediately:
first review whether the additional channels improve the performance tail
enough to justify doubled input acquisition and detector refitting. January
remains closed. Target profiling and prospective validation remain required
before deployment.

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
- `experiments/active/phase5a_64channel_width_comparison.py`
- `models/indy_32ch/64x64checkpoint.pt`
- `models/indy_32ch/48x48checkpoint.pt`

Everything under `history/` is provenance only and must not be imported.
Completed detector runners and regression checks are in `history/phase3/`;
all completed architecture studies and the one-run 48/48 builder are in
`history/phase4/`.
