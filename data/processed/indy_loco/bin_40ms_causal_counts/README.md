# 40 ms causal spike counts

Recommended processed source for the next 32-channel candidate. Each output NPZ
contains unsmoothed 40 ms spike counts for all available electrodes and the 3D
primary-fingertip velocity target.

No centered Gaussian, zero-phase filtering, central difference, or EWMA is baked
into the neural input. The velocity target uses a forward-only low-pass and
backward difference. Causal EWMA is applied later from explicit model configuration.

```bash
python data/processing/indy_loco/build_bin_40ms_causal_counts.py
python data/processing/indy_loco/build_bin_40ms_causal_counts.py \
  --sessions indy_20160915_01 indy_20160916_01
```

Generated NPZ files and run manifests are ignored. The processing source lives in
`data/processing/`; this folder contains only outputs and their format documentation.
