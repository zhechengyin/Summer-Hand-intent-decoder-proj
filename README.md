# Neural Intent Decoder

Compact causal TCN+GRU decoding experiments for intracortical monkey recordings,
with STM32 deployment constraints.

## Start here

- [`docs/STATUS.md`](docs/STATUS.md) — the single authoritative current status.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — ordered next steps.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — active engineering decisions and caveats.
- [`data/README.md`](data/README.md) — where newly downloaded, raw, interim, and
  processed data belong.

## Repository map

```text
src/intent_decoder/       reusable data, feature, model, training and drift code
configs/                  versioned dataset/model/deployment configurations
models/                   checkpoints and model-specific deployment artifacts
experiments/
  active/                 experiments that may change a current decision
  deepblue/               separate U-M finger-SBP benchmark
data/
  raw/                    verified immutable source recordings
  processed/              documented model-ready datasets
  processing/             dataset-specific Python conversion scripts
docs/history/             historical summary and chronological experiment log
history/                  superseded executable experiments; never imported by active code
results/metrics/          small versioned JSON evidence
results/large/            large regenerated logs/figures, ignored by Git
```

## Current truth

Most historical checkpoint and legacy code was removed after its outcomes were
preserved in [`docs/history/`](docs/history/). The few recent executable files
needed to explain the sampling decision are isolated under [`history/`](history/README.md)
and are never imported by active code.

The 32-channel counts-plus-causal-EWMA pipeline is the current research candidate.
Its code now also uses forward-only target filtering, backward differences,
past-only prefix normalization and a strictly causal model. All earlier scores
were superseded by the corrected pipeline. All chronological training and
validation sessions have now been evaluated across CPU seeds 42/43/44, and
session-balanced training is the frozen sampler. A self-contained Phase-1 Optuna
entry point is ready but has not been run. The project still needs that
validation-only optimization, an independently validated drift threshold, one
locked January test evaluation, checkpoint promotion, int8 export, and MCU timing.
