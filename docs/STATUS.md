# Project Status

Updated: 2026-07-30

## Current state

FingerMovements is the active dataset. The processed input contains 316
official training cases and 100 locked test cases, each with 28 EEG channels
and 50 samples. The task is binary prediction of left- versus right-hand finger
movement.

Phase 1b is complete. It compared a feature-linear classifier, Tiny MLP, Tiny
EEGNet, and Tiny multi-scale CNN using seeds 42, 43, and 44 with stratified
five-fold cross-validation. All 60 fits completed, each fold learned
normalization from its training subset only, and the official test file was not
loaded.

Feature + Linear is the frozen Phase-1b baseline: 58.65% mean out-of-fold
accuracy, 1.56 percentage-point seed standard deviation, and 394 trainable
parameters. Tiny EEGNet is retained as the neural firmware candidate: 56.96%
mean accuracy, the lowest seed standard deviation at 0.84 points, and 1,050
parameters. No checkpoint has been selected or trained on all 316 cases.

## Indy archive

All Indy-specific material has been moved to `history/indy/`, including:

- the 37 raw sessions and all processed arrays;
- processing notebook and causal target code;
- dataset, model, and detector configurations;
- model, detector, runtime, sampling, and feature code;
- retained 32-channel checkpoints and Phase-5a experiment checkpoints;
- Phase 0 through Phase 5 experiment code and results;
- causality and sampling tests.

The pre-reset status and experiment log are preserved as
`history/indy/STATUS.md` and `history/indy/EXPERIMENT_LOG.md`. The archive is
provenance only and must not be imported by new active code.

## Next gate

Keep the official test split locked. Before broad architecture sweeping, define
a small Phase 1c comparison of the current feature-linear pipeline, Tiny
EEGNet, and one firmware-compatible spatial feature/classifier such as a
fold-trained CSP- or Fisher-style projection. Any epoch or hyperparameter
selection must use inner validation rather than the outer reporting fold.

## Supported active files

- `README.md`
- `docs/STATUS.md`
- `docs/history/EXPERIMENT_LOG.md`
- `data/README.md`
- `configs/README.md`
- `models/README.md`
- `experiments/active/README.md`
- `results/README.md`
- `history/README.md`
- `history/indy/README.md`
