# Project Status

Updated: 2026-07-29

## Current state

The repository has been reset for a new deployment-first research direction.
There is no active dataset, model, checkpoint, frozen hyperparameter set, or
experiment result.

The goal is to build several small, independent models for different neural
signal prediction or classification tasks. Each candidate must provide useful
task performance while remaining practical for low-latency firmware inference.

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

Before training begins, write one task contract containing:

1. signal modality, channels, sampling rate, and firmware availability;
2. exact label meaning and output shape;
3. dataset unit of independence, split policy, and locked test policy;
4. baseline metrics for the task;
5. firmware budgets for parameters, RAM, Flash, and per-sample latency.

The first experiment should compare a linear baseline and one or two tiny
nonlinear models under the same split. Architecture sweeps start only after
that baseline is reproducible.

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
