# Promoted Phase 6 Indy Model

This folder contains the current best Indy validation candidate and its exact,
standalone inference architecture.

| Item | Value |
|---|---|
| Checkpoint | `phase6_96ch_64x64_checkpoint.pt` |
| Physical inputs | 96 channels |
| Model input | 96 raw counts + 96 causal EWMAs |
| Temporal context | 50 past 40 ms bins (2 seconds) |
| Architecture | four-block causal TCN, width 64; unidirectional GRU, width 64 |
| Parameters | 86,978 |
| Training regularization | model dropout 0.10; paired physical-channel dropout 0.20 |
| Optimizer settings | LR 0.0009; WD 0.025; session-balanced sampling |
| Promoted run | seed 43, epoch 15 |
| December validation | pooled R² 0.7022; macro R² 0.7041; worst session R² 0.6029 |
| SHA-256 | `685ee659b56e40d2484d09b4d03bbdcb032856e772228fb0125c3703575e378a` |

The promoted run was selected by minimum pooled December validation loss. The
configuration was confirmed over seeds 42–44: pooled R² `0.7004 ± 0.0019` and
macro R² `0.7023 ± 0.0015`. January was never loaded by Phase 6.

Paired channel dropout was a training-only regularizer. It removed each raw
channel and its corresponding EWMA together; it is inactive during inference.
The checkpoint also stores the training-derived feature floors, target scaling,
channel order, session lists, training history, and per-session validation
metrics needed to reproduce the selected state.

This validation-selected model supersedes the 32-channel models as the strongest
accuracy candidate. The smaller checkpoints under `../indy_32ch/` remain
available as firmware-size references and are not overwritten.
