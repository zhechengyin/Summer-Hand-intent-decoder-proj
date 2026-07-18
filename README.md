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
results/metrics/          small versioned JSON evidence
results/large/            large regenerated logs/figures, ignored by Git
```

## Current truth

Historical checkpoints and executable archive/legacy code were removed after
their outcomes were preserved in
[`docs/history/ARCHIVE_RETIREMENT.md`](docs/history/ARCHIVE_RETIREMENT.md).

The 32-channel counts-plus-causal-EWMA pipeline is the current research candidate.
Its code now also uses forward-only target filtering, backward differences,
past-only prefix normalization and a strictly causal model. All earlier scores
must be rerun. It still needs full-session-pool evaluation, an independently
validated drift threshold, checkpoint, int8 export, and MCU timing before promotion.
