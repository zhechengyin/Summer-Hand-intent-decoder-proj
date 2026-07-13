# Models

Each model family owns one snake-case folder containing everything required to
understand, train, evaluate, and deploy it.

```text
models/
  tcn_gru/
    best_model.py       readable architecture source
    config.py           preprocessing and model configuration
    data_split.json     recording split provenance
    evaluate.py         train/validation/test evaluation
    train_and_save.py   checkpoint training entry point
    checkpoint.pt       learned weights and normalization metadata
    README.md            results, assumptions, and commands
```

New model families must use separate sibling folders, such as `ewma_mlp/` or
`channel_aware_tcn/`. Do not place architecture-specific source or checkpoints
at the `models/` root, and do not share mutable configuration between model
folders. This keeps comparisons reproducible and prevents one experiment from
silently changing another model.

## Families

- [`tcn_gru/`](tcn_gru/) — full 96-channel reference decoder (TCN+GRU, 0.77 MB).
  Offline/high-channel ceiling.
- [`tcn_gru_8ch/`](tcn_gru_8ch/) — **deployment model of record**: **8 channels /
  STM32**, **strictly causal** (real-time), 24 sessions → **TEST R² 0.606, ~73 KB
  int8, 0 ms lookahead**. (Bidirectional 0.677 is an offline reference only —
  not deployable.) The target for the current 8-channel spike-detection hardware.
  Reuses `tcn_gru`'s `build_net` with `bidir=False`.

Active R²-improvement experiments live in [`../research/`](../research); archived
sweeps in [`../legacy/`](../legacy).
