# Data

FingerMovements is the active dataset.

```text
data/
  raw/FingerMovements/                    immutable downloaded archive
  processed/finger_movements/             train.npz, test.npz, and schema README
  processing/finger_movements/            supported conversion code
```

Rules:

1. Never modify source files under `raw/`.
2. Preserve subject, session, trial, and time boundaries supplied by the source.
3. Define train, validation, and locked test splits before model tuning.
4. Keep processing code under `processing/`, not beside generated arrays.
5. Document every processed schema and unit conversion.
6. Store only inputs that can be reproduced by the intended firmware unless a
   field is explicitly marked as an offline label or audit field.

The FingerMovements converter reads the official BCI Competition II 100 Hz
MATLAB release, preserves its train/test split, and does not normalize, filter,
or augment the EEG. See
`processed/finger_movements/README.md` for the exact NPZ schema.

Indy Loco data is isolated in the sibling project at `indy_loco/data/`.
