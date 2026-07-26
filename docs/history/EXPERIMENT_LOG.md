# Experiment Log

Only experiments that determine the current data pipeline, frozen model or next
research direction are retained here. Superseded model families, exploratory
calibration attempts and abandoned sweep branches were intentionally removed.

## Phase 0a — Causal Indy dataset and month audit

All 37 official Indy sessions were verified against the Zenodo checksum
inventory and converted to `indy_counts_velocity_v2`.

The supported pipeline removed every known future-data dependency:

- centered Gaussian filtering was removed from neural inputs;
- kinematics use the latest already-observed sample at each bin end;
- velocity uses a forward-only low-pass filter and backward difference;
- feature normalization uses only the first 60 seconds of the current session;
- warm-up outputs are discarded;
- the TCN crops right padding and the GRU is unidirectional.

The chronological split is 29 train / 4 December validation / 4 January test.
The audit passed all checksum, provenance, schema, shape and finite-value checks.
It also found real neural drift: selected-32 month classification reached 72.97%
versus a 16.95% permutation null, while target kinematics drifted less strongly.

Decision: retain every technically valid session, preserve session boundaries,
use a training-derived variance floor, and report per-session as well as pooled
metrics.

Evidence:

- `results/indy/phase0a_data_audit/phase0a_data_audit_metrics.json`
- `results/indy/phase0a_data_audit/phase0a_data_audit_session_quality.csv`
- `results/indy/phase0a_data_audit/phase0a_data_audit_month_summary.csv`
- `results/indy/phase0a_data_audit/phase0a_data_audit_month_pairwise.csv`

## Phase 0b — Training sampler selection

Window-, session- and month-balanced training were compared under identical
budgets on CPU seeds 42, 43 and 44.

| Sampler | Validation loss, mean +/- SD | Validation R², mean +/- SD | Seed wins |
| --- | ---: | ---: | ---: |
| Window-balanced | 0.5329 +/- 0.0112 | 0.5080 +/- 0.0113 | 0/3 |
| Session-balanced | **0.5074 +/- 0.0221** | **0.5342 +/- 0.0198** | **3/3** |
| Month-balanced | 0.5259 +/- 0.0066 | 0.5183 +/- 0.0077 | 0/3 |

Decision: freeze session-balanced sampling. The model-specific sampler now
lives in `models/indy_32ch/sampling.py`; the old training scripts were deleted.

Evidence:
`results/indy/phase0b_sampler_selection/phase0b_sampler_selection_metrics.json`.

## Phase 1a–1e — Hyperparameter selection

Phase-1 through Phase-1e used December validation only; January was not loaded.
The sequence was:

1. a 40-trial Optuna search to locate the learning-rate/regularization region;
2. equal-budget dropout and weight-decay boundary grids;
3. five-seed confirmation of the two remaining weight-decay candidates.

Five-seed final comparison:

| Weight decay | Validation loss, mean +/- SD | Pooled R², mean +/- SD | Macro R² | Worst-session R² | Wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.025 | 0.497354 +/- 0.015236 | 0.540861 +/- 0.015925 | 0.552665 | 0.269018 | 2/5 |
| 0.060 | **0.494178 +/- 0.016946** | **0.543641 +/- 0.017181** | **0.555413** | **0.275041** | **3/5** |

Weight decay 0.060 passed the preregistered rule: at least 3/5 wins, at least
0.003 mean-loss improvement, and no macro/worst-session R² degradation.

Decision: freeze session-balanced sampling, learning rate 0.0009, weight decay
0.060, dropout 0.025, batch size 32, 20-epoch budget and the current causal
architecture. Seed 43 epoch 7 was selected before test because it had the best
loss and pooled R² among the frozen-candidate checkpoints.

The completed scripts and final five-seed JSON are under `history/phase1/`.
Intermediate databases, figures, checkpoints and duplicate JSON files were
deleted. The only retained model artifact is
`models/indy_32ch/checkpoint.pt`.

## Phase 2 — Locked January evaluation

Completed: 2026-07-22 20:38:30 UTC.

The frozen seed-43 checkpoint ran once on all four January sessions. The script
created no optimizer, performed no backward pass, changed no weight and used
labels only after inference for scoring.

| Session | Windows | Loss | R² x | R² y | Mean R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| `indy_20170123_02` | 274 | 0.305255 | 0.679469 | 0.709934 | 0.694701 |
| `indy_20170124_01` | 265 | 0.874137 | -0.177063 | 0.072258 | -0.052402 |
| `indy_20170127_03` | 336 | 0.318691 | 0.661649 | 0.736107 | 0.698878 |
| `indy_20170131_02` | 377 | 0.282975 | 0.636575 | 0.719383 | 0.677979 |

Aggregate:

- pooled loss: 0.422562;
- pooled R²: 0.551146 versus validation 0.560362;
- session-macro R²: 0.504789;
- three non-failing sessions averaged R² 0.690520;
- worst-session R²: -0.052402.

`indy_20170124_01` is technically valid but its fixed-32 prefix differs sharply
from training: log-rate correlation -0.059, robust distance 3.910, two silent
fixed channels and median rate ratio 0.313. Successful January sessions had
correlations 0.362--0.555, distances 1.149--1.488 and no silent fixed channel.
These post-hoc values explain the failure direction but cannot select a
detector threshold.

Decision: January is consumed. Do not compare another model on it. The next
experiment is a label-free drift gate developed only on the 33 pre-January
sessions. Its first action is decode versus flag/abstain, not adaptation.

Evidence:

- `results/indy/phase2_locked_january/phase2_locked_january_metrics.json`
- `results/indy/phase2_locked_january/phase2_locked_january_figure.png`
- historical runner: `history/phase2/phase2_locked_january.py`

## Phase 3a — Pre-January label-free drift-detector baseline

Completed: 2026-07-24.

The first detector implementation uses only the 29 train and four December
validation sessions. It reads the first 1,500 fixed-channel count bins, never
loads velocity, rejects January-or-later names in code, and compares:

1. multi-month log-rate correlation, robust rate distance, global rate ratio
   and unexpectedly silent channels;
2. one global five-dimensional Gaussian KLD;
3. the nearest of several month-reference five-dimensional Gaussian KLDs;
4. the interpretable features and multi-reference KLD together.

The KLD baseline is inspired by MINDFUL but is intentionally project-specific:
PCA and normalization are fitted only from historical references, monthly
prototypes replace the paper's performance-selected initial reference, and no
self-supervised correction is performed.

The first 0.95-quantile run showed that session-level calibration was
over-optimistic for a completely unseen month. Calibration was corrected to an
inner leave-one-complete-month-out loop. A conservative 0.99 empirical
quantile, selected using pre-January data only, reduced intact-session flags to
5/33 and abstentions to 3/33.

| Condition | Combined flag | Combined abstain |
| --- | ---: | ---: |
| Intact held-month session | 15.2% | 9.1% |
| 50% global thinning | 18.2% | 6.1% |
| 75% global thinning | 72.7% | 45.5% |
| 25% channel dropout | 100.0% | 18.2% |
| Channel permutation | 18.2% | 3.0% |
| 65% thinning + 25% dropout | 100.0% | 72.7% |

Global KLD and multi-reference KLD each flagged 1/33 intact sessions, but global
KLD was more sensitive to most synthetic faults. Therefore the multi-reference
MINDFUL variant is retained as a baseline, not promoted as the winner.

Decision: the implementation is valid enough for the next experiment, but the
9.1% intact-data abstain rate is too high to freeze without performance labels.
Phase 3b must pair these scores with strictly out-of-month decoder R² and
determine whether the three abstained sessions were actually incompatible.
Synthetic faults are engineering checks, not evidence that real decoder
failures can be predicted.

Evidence:

- `results/indy/phase3a_drift_detector/phase3a_drift_detector_metrics.json`
- `results/indy/phase3a_drift_detector/phase3a_drift_detector_scores.csv`
- `results/indy/phase3a_drift_detector/phase3a_drift_detector_figure.png`
- `results/indy/phase3a_drift_detector/phase3a_drift_detector_reference.npz`

## Phase 3b — Strict leave-one-month-out decoder evaluation

Completed: 2026-07-25.

Five temporary decoders were trained, each excluding one complete pre-January
month. Architecture, selected channels, session-balanced sampling, seed 43,
epoch 7, optimizer and scheduler trajectory were fixed before each fold. Held
velocity remained unloaded until the optimizer had finished.

| Held month | Session-macro R² | Worst session R² |
| --- | ---: | ---: |
| 2016-04 | 0.418367 | 0.232095 |
| 2016-06 | 0.398743 | -0.136478 |
| 2016-09 | 0.735912 | 0.687347 |
| 2016-10 | 0.592236 | -0.054134 |
| 2016-12 | 0.570223 | 0.314350 |

Across all 33 held sessions, macro R² was 0.559710 and pooled R² was 0.526599.
The two negative sessions were `indy_20160630_01` and
`indy_20161013_03`. The raw-count detector caught the October failure but
passed the worse June failure, so its three-level thresholds were not promoted.

Evidence:

- `results/indy/phase3b_leave_one_month_out/phase3b_leave_one_month_out_metrics.json`
- `results/indy/phase3b_leave_one_month_out/phase3b_leave_one_month_out_sessions.csv`
- `results/indy/phase3b_leave_one_month_out/phase3b_leave_one_month_out_folds.csv`
- `results/indy/phase3b_leave_one_month_out/phase3b_leave_one_month_out_figure.png`

## Phase 3c — Frozen-decoder hidden/output compatibility layer

Completed: 2026-07-25.

The second layer runs the frozen outer-fold decoder on only the first 60
seconds, then compares reference-only distributions of GRU hidden states,
predicted output and output differences. It accepts no velocity label and
never creates an optimizer.

The first broad rule allowed warning-level hidden, output-delta and temporal
evidence to vote. It caught both failures but also abstained
`indy_20160622_01` and `indy_20161206_02`, while temporal evidence warned three
additional usable sessions. Those scores remain recorded but cannot veto.

The retained conservative rule requires both:

1. hidden-state KLD above its nested 0.99 severe threshold;
2. absolute predicted-output KLD above its nested 0.99 severe threshold.

Under this rule both negative-R² sessions abstained and the other 31 sessions
passed. The result remained unchanged for all nine combinations of hidden PCA
dimensions 3/5/8 and covariance shrinkage 0.05/0.10/0.20.

Decision: integrate the two-layer wrapper and fitted reference artifacts as a
development candidate. Do not describe this as independent validation because
the known Phase-3b outcomes selected the retained rule. January remains
forbidden; prospective sessions and STM32 equivalence are required before
deployment freezing.

Evidence:

- `results/indy/phase3c_decoder_state_detector/phase3c_decoder_state_detector_metrics.json`
- `results/indy/phase3c_decoder_state_detector/phase3c_decoder_state_detector_sessions.csv`
- `results/indy/phase3c_decoder_state_detector/phase3c_decoder_state_detector_sensitivity.csv`
- `results/indy/phase3c_decoder_state_detector/phase3c_decoder_state_detector_figure.png`
- `results/indy/phase3c_decoder_state_detector/phase3c_active_gate_metadata.json`

## Phase 3 integration and archival

Completed: 2026-07-26.

The experimental runners and regression files from Phase 3a--3c were moved to
`history/phase3/`. They are provenance and are not imported by active code.

`models/indy_32ch/runtime.py` now provides the single integrated execution
order:

1. receive the frozen 32-channel count mapping;
2. collect the first 1,500 bins without releasing predictions;
3. run the raw-count and frozen-decoder-state detector layers;
4. combine their pass/warning/abstain decisions;
5. block on abstain, otherwise decode only bins after the 60-second prefix.

The saved detector classes now support non-pickle artifact loading and validate
reference months, channel mapping, warm-up length and checkpoint compatibility.
An end-to-end CPU smoke test on `indy_20161207_02` returned pass and released
9,600 post-warm-up prediction bins. All 13 Phase-3 regression tests passed
before archival.

The final compatible reference fit excludes the two Phase-3b negative-R²
sessions, leaving 31 references. A required artifact-level audit exposed a
scope difference that must remain visible:

| Session | Strict held-month detector | Final active-checkpoint artifact |
| --- | --- | --- |
| `indy_20160630_01` | abstain | pass |
| `indy_20161013_03` | abstain | abstain |

This is not a loader bug. The strict June fold used a decoder trained without
June, while the retained active checkpoint had already learned from June 30.
Consequently, the Phase-3c two-failure result remains valid for its strict
out-of-month protocol, but it cannot be copied directly into a claim about the
final active artifact.

Decision: keep the integrated runtime as a transparent development candidate.
Do not tune it with January, do not claim that it catches every incompatible
session, and require prospective data before deployment freezing.
