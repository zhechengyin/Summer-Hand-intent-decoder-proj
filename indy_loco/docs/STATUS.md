# Indy Loco Status

**State:** Phase 9 is complete. For the promoted Phase 6 causal checkpoint, a
continuous rolling 50-bin past-window reached pooled December validation R²
`0.7526`, compared with `0.7021` for the original 50-bin block-reset policy.
The rolling policy won all four validation sessions, was frozen, and then
reached pooled January R² `0.7277` in a single locked-test pass. January was not
used to select the policy. The Indy half of Phase 8 is also complete: 48 ms lookahead reached test
R² `0.7576 ± 0.0396` and 100 ms reached `0.7554 ± 0.0397` over 15 folds each.
The equivalent 30-fit Loco runner is ready but has not been trained. Phase 7
remains complete at test R² `0.7056 ± 0.0722`. The Phase 6 96-channel 64/64
TCN+GRU remains the strongest general validation-selected candidate, and all
retained checkpoints are intact. For presentation and deployment planning,
the retained work is now organized into Tiny, Mid-size, and Large operating
tiers. These tiers deliberately do not use channel count as their defining
label.

## Phase 9 deployment-policy replay

Phase 9 did not retrain or modify the Phase 6 checkpoint. It preserved the
96-channel raw-plus-causal-EWMA preprocessing and session-local 60-second
calibration, then compared two inference policies on December validation only:

- **A — block reset:** start at bin 1500, zero-fill the unseen suffix of each
  non-overlapping 50-bin block, and clear the block every two seconds;
- **B — rolling calibration seed:** at 60 seconds, normalize calibration bins
  1450–1499 with the newly frozen statistics, then keep a continuous stride-1
  past-only window without clearing it.

| Validation aggregation | A mean R² | B mean R² | A MSE | B MSE |
|---|---:|---:|---:|---:|
| Pooled, first 10 s | 0.7278 | **0.8059** | 8.2333 | **5.6674** |
| Pooled, first 30 s | 0.7144 | **0.7690** | 10.1647 | **8.1631** |
| Pooled, all post-calibration | 0.7021 | **0.7526** | 14.8720 | **12.3567** |
| Per-session macro, all post-calibration | 0.7039 | **0.7540** | 14.6735 | **12.1699** |

Strategy B improved full-session mean R² by `0.0505` and reduced pooled MSE
by about `16.9%`. It also won every validation session. Short-interval
per-session macro R² is not used for selection because one session has almost
no target variance in its opening interval, making R² numerically unstable;
pooled R², MSE, and full-session per-session results agree on B.

After B was frozen, the runner loaded January and evaluated B only. Locked-test
pooled mean R² was `0.7277` with MSE `12.3149`. The replay also verified A
against full-block inference to within `2.9e-6` maximum absolute error. Phase 9
therefore recommends seeding firmware with the final two calibration seconds
and never clearing the rolling inference window. Exact replay artifacts and
golden vectors are under `results/phase9_deployment_policy_replay/`.

## Tiny, Mid-size, and Large model tiers

The three tiers describe deployment operating points rather than three claims
of directly comparable accuracy. Tiny and Mid-size report chronological
December validation results, while Large reports Phase 8 reach-level five-fold
test results on three Indy benchmark sessions. Their R² values must therefore
retain the protocol labels below and must not be plotted as one controlled
architecture sweep.

| Tier | Current architecture | Temporal contract | Parameters | Current result | Deployment role |
|---|---|---|---:|---|---|
| **Tiny** | Four-block causal TCN, width 48; one-layer unidirectional GRU, width 48; linear 2-D velocity head | 50 past 40 ms bins; no future data | 45,266 | December pooled validation R² **0.5651**; macro R² 0.5750; worst-session R² 0.3461 | Smallest retained firmware candidate; lowest compute and memory demand, intended for minimum-latency on-chip deployment |
| **Mid-size** | Four-block causal TCN, width 64; one-layer unidirectional GRU, width 64; linear 2-D velocity head; paired channel dropout used only during training | 50 past 40 ms bins; strictly causal, no future data | 86,978 | Promoted seed-43 December pooled validation R² **0.7022**; three-seed pooled R² **0.7004 ± 0.0019** | Primary real-time decoder, balancing causal latency and accuracy; current strongest general validation-selected checkpoint |
| **Large** | Phase 8 lookahead TCN+GRU: the Mid-size 64/64 topology retrained per reach fold with future-aligned neural input | Same past context plus **48 ms neural lookahead**; deliberately non-causal | 86,978 in the completed experiment | 15-fold Indy test R² **0.7576 ± 0.0396** at 48 ms; 100 ms reached 0.7554 ± 0.0397 | Highest-performing operating mode; trades buffering and at least 48 ms of algorithmic latency for accuracy |

The Tiny tier is **not an MLP**: the project has no retained, validated Indy
MLP checkpoint. The current Tiny evidence is the compact 48/48 causal TCN+GRU.
The Large tier is currently larger at the **system level** because it requires
future-sample buffering and accepts longer latency, but its trained neural
network has the same 86,978-parameter topology as Mid-size. Off-chip memory is
therefore an available implementation option, not a requirement established by
the present model. A professor requirement for a structurally larger third
network or mandatory off-chip execution would require a new approved training
and deployment experiment; it should not be claimed from Phase 8 alone.

Recommended presentation labels are:

- **Tiny — Compact causal / minimum resource**
- **Mid-size — Full causal / real-time balanced**
- **Large — Lookahead / maximum measured accuracy**

## Active Phase 8 protocol

`experiments/active/phase8_future_lookahead_fivefold.py` used the three Indy
paper sessions and identical reach folds for both lookahead conditions. Every
fold starts from fresh seed-43 weights. All feature/target statistics and model
weights use training reaches only; validation selects a checkpoint, and test is
evaluated afterward. The model, optimizer, 96-channel input, and paired channel
dropout are frozen from Phase 6 so lookahead is the controlled variable.

The nominal 50 ms condition used 12 native samples = 48 ms, never 52 ms,
because 52 ms would exceed the permitted future interval. The 100 ms condition
uses exactly 25 native samples. This experiment intentionally trades latency
for accuracy and must not be described as strictly causal.

The completed Indy session means were:

| Condition | `20160622` | `20160630` | `20170131` | Overall |
|---|---:|---:|---:|---:|
| 48 ms | 0.8046 | 0.7225 | 0.7457 | **0.7576** |
| 100 ms | 0.8028 | 0.7343 | 0.7289 | 0.7554 |

`experiments/active/phase8_loco_future_lookahead_fivefold.py` now repeats the
same protocol for the three Loco sessions. Every fold selects 96 of 192 source
channels using training reaches only, freezes that selection across both
lookahead conditions, and retains 0.20 paired channel dropout. No Loco Phase 8
R² has been reported yet.

## Final Phase 7 benchmark

The exact Loco sessions used by the NeuroBench primate-reaching benchmark are
downloaded from Zenodo and MD5-verified: `loco_20170210_03`,
`loco_20170215_02`, and `loco_20170301_05`. They are processed independently at
the native 4 ms interval into 192-channel binary spike-presence arrays. Unit
slots are combined per physical channel with the same logical-OR rule as the
official loader.

Each NPZ also preserves cursor/target kinematics, original timestamps, reach
boundaries, and the official ordered reach split: first 50% train, next 25%
validation, and final 25% test. No model statistics, normalization, or weights
are fit during conversion. The official central-gradient velocity target is
retained for benchmark comparability and is explicitly separate from a
deployment-focused causal-target claim.

`history/experiments/phase7/phase7_ann_vs_snn_fivefold.py` trained a separate fresh
seed-43 decoder for each session and fold. All 96 Indy channels were used. Each
Loco fold selected 96 of 192 channels from training reaches only and retained
the winning 0.20 paired channel dropout. Train-only statistics and weights did
not use held-out reaches; validation selected each checkpoint and test was
evaluated only afterward.

| Session | Five-fold test R² |
|---|---:|
| `indy_20160622_01` | 0.8066 ± 0.0153 |
| `indy_20160630_01` | 0.6828 ± 0.0115 |
| `indy_20170131_02` | 0.7443 ± 0.0445 |
| `loco_20170210_03` | 0.6749 ± 0.0289 |
| `loco_20170215_02` | 0.6277 ± 0.0838 |
| `loco_20170301_05` | 0.6974 ± 0.0590 |
| **All 30 folds** | **0.7056 ± 0.0722** |

The overall mean is 0.0395 above the paper's SNN3D reference of 0.6661. This
is a protocol-level benchmark rather than a perfectly controlled architecture
comparison: Phase 7 used the project's 40 ms causal TCN+GRU and restricted
Loco to 96 selected inputs. Indy averaged 0.7446 across its 15 folds and Loco
averaged 0.6667. The weakest fold was 0.5211, and
`loco_20170215_02` had the highest session-level SD (0.0838), so cross-reach
stability remains uneven despite the strong overall result.

The 30 fold checkpoints are retained under
`history/results/indy/phase7_ann_vs_snn_fivefold/checkpoints/` as
reproducibility evidence.
They are session-specific benchmark models and do not replace the promoted
Phase 6 firmware candidate.

## Final Phase 6 result

The controlled runner kept the 64/64 width, learning rate 0.0009, weight decay
0.025, model dropout 0.10, causal preprocessing, and session-balanced sampler
fixed. It screened channel counts/rankings, kernel size 2, a three-block TCN,
and paired channel dropout. Seed 43 performed the screen; fixed references and
category winners were confirmed with seeds 42 and 44.

The runner automatically selects NVIDIA CUDA when available and otherwise uses
CPU. Apple MPS remains disabled because it previously produced invalid training
gradients for this model graph.

All 29 train sessions updated weights. Channel ranking used only their first
60-second prefixes and includes activity, cross-session stability, availability,
and drift across training months. December was inference-only and selected
configurations/checkpoints. January was never loaded. See
`experiments/active/README.md` for the exact command and protocol.

The winner used all 96 channels, kernel size 3, four TCN blocks, and 0.20 paired
channel dropout. Across seeds 42–44 it achieved pooled December validation R²
`0.7004 ± 0.0019`, macro R² `0.7023 ± 0.0015`, mean worst-session R²
`0.6085`, and minimum worst-session R² `0.5950`. Its seed-specific pooled R²
values were 0.6978, 0.7022, and 0.7012.

The no-channel-dropout 96-channel reference reached only pooled R²
`0.6476 ± 0.0057`. The improvement therefore came from combining the wider
input with channel-level regularization, not from channel count alone.

## Data and protocol

| Item | Final setting |
|---|---|
| Sessions | 29 train / 4 December validation / 4 January test |
| Bin and window | 40 ms bins; 50-bin (2 s) past-only windows |
| Promoted Phase 6 input | 96 raw counts + 96 causal EWMA features |
| Target | x/y fingertip velocity |
| Calibration | first 60 s of each session; past-only |
| Sampling | session-balanced |
| Optimizer settings | learning rate 0.0009; weight decay 0.025; model dropout 0.10; paired channel dropout 0.20; batch 32 |

## Initial Phase 6 diagnostic

The all-96 seed-43 CUDA run selected epoch 5. Train R² was 0.8283 and pooled
December validation R² was 0.6439 (macro 0.6466; worst session 0.5444). By epoch
20, train R² reached 0.8821 while validation R² was 0.6202. The checkpoint rule
protected the better epoch-5 state, but the widening gap motivated the
controlled sweep. CPU and CUDA reruns reportedly showed the same qualitative
overfitting pattern; the completed sweep forbade mixing backends within one
resumable result set.

## Retained checkpoints

| Checkpoint | Role | Parameters | Validation result | SHA-256 |
|---|---|---:|---|---|
| `models/indy_96ch/phase6_96ch_64x64_checkpoint.pt` | strongest validation candidate | 86,978 | pooled R² 0.7022; macro 0.7041; worst 0.6029 | `685ee659b56e40d2484d09b4d03bbdcb032856e772228fb0125c3703575e378a` |
| `models/indy_32ch/48x48checkpoint.pt` | preferred standalone firmware candidate | 45,266 | pooled R² 0.5651; macro 0.5750; worst 0.3461 | `5c8b375787ff93f90006df5f0cfea07303660928c7b69a84d4d75e1a368319ef` |
| `models/indy_32ch/64x64checkpoint.pt` | reference model used by detector work | 78,786 | pooled R² 0.5604; macro 0.5702; worst 0.3144 | `2ee52c426ee43ba88cebe7c85dd8392f40f9e75748abe9bbf4e94093556363a5` |

The 48/48 checkpoint was trained with seed 43 for 20 epochs and selected epoch 10 by December validation loss. It is not interchangeable with the detector artifacts built from 64/64 hidden states.

## Strongest evidence

- Sampler comparison: session-balanced sampling won all three seeds over window- and month-balanced sampling.
- Five-seed architecture comparison: 48/48 passed the predefined non-inferiority limits against 64/64 and used 42.5% fewer parameters.
- Strict pre-January leave-one-month-out evaluation of 64/64: macro R² 0.5597 and pooled R² 0.5266. June and October contained severe failures, showing real cross-session drift risk.
- January one-shot evaluation: pooled R² 0.5511; three sessions were strong, while `indy_20170124_01` had R² -0.0524.

## Detector status

The final prototype combined a 60-second label-free signal check with frozen-decoder hidden/output compatibility scores. In retrospective leave-one-month-out analysis, the second layer identified the June 30 and October 13 failures without rejecting the other 31 sessions across nine sensitivity settings.

This does not establish prospective reliability. The integrated 64/64 artifact passed June when the checkpoint had trained on June, demonstrating that detector behavior depends on the exact frozen decoder and training domain.

## Phase 5a exploratory 64-channel result

The final authoritative run used CPU, seed 43, 30 epochs, training-only fitting, and December validation selection. January was not loaded.

| Architecture | Parameters | Selected epoch | Validation pooled R² | Macro R² | Worst-session R² |
|---|---:|---:|---:|---:|---:|
| 64-channel 64/64 | 82,882 | 7 | 0.6625 | 0.6669 | 0.5842 |
| 64-channel 48/48 | 48,338 | 4 | 0.6569 | 0.6608 | 0.5760 |

The 64/64 option ranked first, but the result was single-seed and was not promoted. An earlier MPS run and an unregistered 32/32 artifact are withdrawn evidence.

## Phase 5 confirmed 64-channel result

Phase 5 swept the 64-channel 64/64 model and compared all 29 training sessions
against a retrospective 27-session detector-filtered policy. January was never
loaded.

| Policy | Hyperparameters | Validation pooled R² | Macro R² | Worst-session evidence |
|---|---|---:|---:|---:|
| All 29 sessions | LR 0.0009; WD 0.025; dropout 0.10 | 0.6575 ± 0.0080 | 0.6627 ± 0.0076 | mean 0.5526; minimum 0.5186 |
| Detector-filtered 27 sessions | LR 0.0009; WD 0.025; dropout 0.10 | 0.6552 ± 0.0034 | 0.6595 | mean 0.5766; minimum 0.5566 |

Filtering improved the lower tail but not the mean. The active training policy
therefore remains all 29 sessions. The detector is not used as a retrospective
training-data cleaner.

## Project-history note

The Indy line was closed in favor of FingerMovements EEG classification on
2026-07-29, then reopened for controlled channel-count and benchmark
experiments. Phases 5–7 are archived and there is no active experiment. The
96-channel candidate remains promoted; all earlier retained checkpoints remain
unchanged for size and detector comparisons.
