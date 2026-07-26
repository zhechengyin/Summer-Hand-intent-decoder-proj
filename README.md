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
- Phase 3b completed strict pre-January leave-one-month-out evaluation:
  session-macro R² was 0.5597 and two held sessions had negative R².
- The active compatibility system is now a two-layer development candidate:
  raw-count checks plus a conservative frozen-decoder hidden/output veto.
- Phase 3c abstained both known negative-R² sessions and passed the other 31
  across nine hidden-dimension/covariance sensitivity settings.
- `models/indy_32ch/runtime.py` is the integrated execution path: it performs
  the 60-second two-layer gate first and releases only later decoder output.
- January remains hard-forbidden for detector development. The runtime is not
  prospectively or deployment-frozen.

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
experiments/active/      reproducible Phase-0a data audit only
results/indy/            phase-aligned Indy metrics and figures
docs/                    current status and concise experiment log
history/                 completed Phase-1, Phase-2 and Phase-3 evidence
```

There is no generic `src/` layer. Dataset-generation code lives with
`data/processing/`; code specific to the only active decoder lives with
`models/indy_32ch/`.

Experiment naming is chronological: Phase 0 covers data/sampler decisions,
Phase 1a--1e covers hyperparameter selection, Phase 2 is the consumed locked
test, and Phase 3 covers label-free drift detection.

Phase 3 is archived. Its strict outer-fold result caught both known
pre-January failures, but the final integrated artifact uses the one active
checkpoint that had already trained on those sessions: it blocks
`indy_20161013_03` but not `indy_20160630_01`. This known limitation is retained
explicitly until future prospective sessions are available.

Do not rerun model selection or compare another checkpoint on the consumed
January test split.
