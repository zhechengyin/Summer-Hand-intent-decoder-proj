# Data organization

Raw recordings and derived datasets are deliberately separated. Never edit raw
files in place and never put generated arrays directly in `source_data/`.

```text
data/
  source_data/
    indy_loco/                 immutable downloaded MAT recordings
      train1.mat ... train6.mat
      eval1.mat
      test1.mat
  processed/
    bin_40ms/
      preprocess.py           exact transformation source
      README.md               method and output contract
      artifacts/              generated NPZ/CSV/manifest files (gitignored)
    bin_50ms/
      preprocess.py
      README.md
      artifacts/
```

Each preprocessing experiment gets its own snake-case folder. That folder must
contain:

1. `preprocess.py` — executable transformation from `source_data/`.
2. `README.md` — parameters, assumptions, command, and artifact schema.
3. `artifacts/` — generated processed data and a run manifest.

Do not combine recording sessions while preprocessing. One output per source
file preserves session boundaries and prevents model windows from crossing from
one recording into another. Large raw and generated artifacts are intentionally
excluded from Git; the code and documentation required to regenerate them are
tracked.

## Adding a technique

Copy the nearest method folder, give it a descriptive name such as
`event_stream_no_bins`, change only its owned script/configuration, document the
output schema, and run it. Models should reference a named processed method
rather than silently changing preprocessing constants.

The split-to-original-session mapping is recorded in
[`../models/tcn_gru/data_split.json`](../models/tcn_gru/data_split.json).
