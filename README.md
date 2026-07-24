# Neural Intent Decoder

Strictly causal 32-channel TCN+GRU decoding for the Indy intracortical reaching
dataset, with STM32 deployment constraints.

## Current state

- All 37 Indy sessions are processed into a fixed chronological 29/4/4 split.
- Input is 32-channel spike counts plus causal EWMA at 40 ms resolution.
- Every session uses a 60-second past-only normalization prefix; warm-up outputs
  are invalid.
- The frozen model is `models/indy_32ch/checkpoint.pt`.
- Frozen hyperparameters are recorded in `configs/indy_32ch.yaml`.
- Locked January pooled R² is 0.5511. Three sessions scored about 0.68--0.70;
  one drifted session scored -0.0524.
- Phase 3a implements a label-free 60-second session compatibility gate using
  interpretable rate checks plus a five-dimensional MINDFUL-inspired KLD.
- The detector uses nested held-month validation and contains a hard code guard
  against loading January. It is implemented but not deployment-frozen.

## Read these files

- [`docs/STATUS.md`](docs/STATUS.md): current truth and next task.
- [`docs/history/EXPERIMENT_LOG.md`](docs/history/EXPERIMENT_LOG.md): concise
  record of the experiments that still affect the current model.
- [`models/indy_32ch/README.md`](models/indy_32ch/README.md): frozen model card.
- [`data/README.md`](data/README.md): data layout and immutability rules.
- [`results/indy/README.md`](results/indy/README.md): phase-aligned result index.

## Repository layout

```text
configs/                 dataset manifest and frozen model configuration
data/
  raw/                   immutable source recordings
  processed/             generated model-ready arrays
  processing/            Indy conversion notebook and causal target transforms
models/indy_32ch/        decoder, detector, input pipeline, sampler and checkpoint
experiments/active/      reproducible Phase-0a and Phase-3a runners
results/indy/            phase-aligned Indy metrics and figures
docs/                    current status and concise experiment log
history/                 completed Phase-1, locked-test and regression evidence
```

There is no generic `src/` layer. Dataset-generation code lives with
`data/processing/`; code specific to the only active decoder lives with
`models/indy_32ch/`.

Experiment naming is chronological: Phase 0 covers data/sampler decisions,
Phase 1a--1e covers hyperparameter selection, Phase 2 is the consumed locked
test, and Phase 3 covers label-free drift detection.

Do not rerun model selection or compare another checkpoint on the consumed
January test split.
