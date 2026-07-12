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

The current model of record is [`tcn_gru/`](tcn_gru/).
