# 40 ms spike-count bins

Current preprocessing baseline. Each raw recording becomes one compressed NPZ
containing:

- `rates`: `float32`, shape `(96, time_bins)`, summed multiunit spike counts per
  electrode, optionally Gaussian-smoothed along time.
- `velocity`: `float32`, shape `(time_bins, 3)`, fingertip velocity.
- `bin_s`, `velocity_lowpass_hz`, and `rate_smoothing_sigma_bins`: scalar metadata.

The script preserves `train*`, `eval*`, and `test*` filenames and writes a JSON
manifest with source/output paths and shapes.

```powershell
py data/processed/bin_40ms/preprocess.py
```

Defaults match the current model: 40 ms bins, 3 Hz position low-pass before
differentiation, sigma=1-bin Gaussian smoothing, first 96 electrodes.
