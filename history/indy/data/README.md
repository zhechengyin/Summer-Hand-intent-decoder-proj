# Data

The active project uses one dataset: Indy.

```text
data/
  raw/indy_loco/indy/                 37 immutable Zenodo MAT files
  processed/indy_loco/indy/
    train/                            29 sessions
    validation/                        4 sessions
    test/                              4 consumed January sessions
  processing/indy_loco/indy/
    prepare_indy_model_ready.ipynb    supported conversion notebook
    causal_targets.py                 causal sample-hold and target velocity
```

Rules:

1. Never modify files under `raw/`.
2. Preserve session boundaries.
3. Generate processed data only through the supported notebook.
4. Do not convert MAT to CSV for model training.
5. Do not use January for future model or detector selection.

To rebuild the model-ready data, open
`processing/indy_loco/indy/prepare_indy_model_ready.ipynb` with the project
virtual environment and run it from top to bottom.

Neural EWMA features and normalization are not preprocessing outputs; they are
computed at model time under `models/indy_32ch/`.

The canonical checksum/split manifest is
`configs/datasets/indy_sessions.yaml`. The model-ready schema is documented in
`processed/indy_loco/indy/README.md`.

Any inactive U-M/Deep Blue files already present under raw or processed storage
are left untouched, but no active code in this repository consumes them.
