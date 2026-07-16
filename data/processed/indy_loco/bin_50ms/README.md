# 50 ms spike-count bins

Alternative preprocessing experiment using 50 ms bins (20 Hz). Its artifact
schema matches `../bin_40ms/` so downstream comparisons can use the same loader.

```powershell
py data/processed/bin_50ms/preprocess.py
```

This folder is a separate method package: outputs never overwrite the 40 ms
baseline.
