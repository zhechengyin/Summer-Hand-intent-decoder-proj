# Data organization

The data tree intentionally uses only two data-storage states plus one code folder:

```text
data/
  raw/                      immutable source recordings
    indy_loco/
      indy/                 37 original Zenodo MAT sessions
    umich_deepblue/
  processed/                generated model-ready datasets
    indy_loco/
      indy/
        train/              29 chronological training sessions
        validation/          4 December validation sessions
        test/                4 locked January test sessions
    umich_deepblue/
  processing/               notebooks/scripts that build processed data
    indy_loco/
      indy/                 structure + preprocessing notebook
    umich_deepblue/
```

## Rules

1. Place source recordings in `raw/` and never edit them in place.
2. Keep processing notebooks/scripts in `processing/`, not beside generated data.
3. Write generated arrays to `processed/`; they must be reproducible from raw data.
4. Each processed method keeps a README describing schema and parameters.
5. Preserve recording/session boundaries; model windows must never cross files.

The canonical Indy loader is `src/intent_decoder/data/indy.py`. Alias/session
provenance and the official 37-file checksum inventory are stored outside the
data tree in `configs/datasets/indy_sessions.yaml`, together with the fixed
chronological split.
