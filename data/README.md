# Data organization

The data tree intentionally uses only two data-storage states plus one code folder:

```text
data/
  raw/                      immutable source recordings
    indy_loco/
    umich_deepblue/
  processed/                generated model-ready datasets
    indy_loco/
      bin_40ms_causal_counts/
    umich_deepblue/
  processing/               Python scripts that build processed data
    indy_loco/
    umich_deepblue/
```

## Rules

1. Place source recordings in `raw/` and never edit them in place.
2. Keep processing Python files in `processing/`, not beside generated data.
3. Write generated arrays to `processed/`; they must be reproducible from raw data.
4. Each processed method keeps a README describing schema and parameters.
5. Preserve recording/session boundaries; model windows must never cross files.

The canonical Indy loader is `src/intent_decoder/data/indy.py`. Alias/session
provenance is stored outside the data tree in `configs/datasets/indy_sessions.yaml`.
