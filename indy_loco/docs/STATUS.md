# Indy Loco Status

**State:** Phase 5 is complete and archived. Phase 6 is active as a controlled
96-channel experiment. Previous retained models remain frozen; no new candidate
has been promoted.

## Active Phase 6 experiment

The active runner trains one 96-channel 64/64 TCN+GRU for 20 epochs using the
Phase 5 winning settings: learning rate 0.0009, weight decay 0.025, dropout 0.10,
and seed 43. It uses all 96 physical channels as 96 raw count streams plus 96
causal EWMAs. No channel ranking is fitted.

The runner automatically selects NVIDIA CUDA when available and otherwise uses
CPU. Apple MPS remains disabled because it previously produced invalid training
gradients for this model graph.

All 29 train sessions update weights under session-balanced sampling. December
is inference-only and selects the checkpoint. January is never loaded. This is
a single-seed channel-count extension, not a promoted model. See
`experiments/active/README.md` for the exact command and protocol.

## Data and protocol

| Item | Final setting |
|---|---|
| Sessions | 29 train / 4 December validation / 4 January test |
| Bin and window | 40 ms bins; 50-bin (2 s) past-only windows |
| Active Phase 6 input | 96 raw counts + 96 causal EWMA features |
| Target | x/y fingertip velocity |
| Calibration | first 60 s of each session; past-only |
| Sampling | session-balanced |
| Active optimizer settings | learning rate 0.0009; weight decay 0.025; dropout 0.10; batch 32; seed 43 |

## Retained 32-channel checkpoints

| Checkpoint | Role | Parameters | Validation result | SHA-256 |
|---|---|---:|---|---|
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
2026-07-29, then reopened for controlled channel-count experiments. Phase 5 is
archived and Phase 6 is active. All earlier retained checkpoints remain
unchanged until a new result is reviewed and explicitly promoted.
