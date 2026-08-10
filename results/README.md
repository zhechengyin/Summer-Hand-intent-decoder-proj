# Results

The active Phase 2c bin/window sweep results are stored at:

```text
results/finger_movements/phase2c_bin_window_sweep/
```

The provisional best window is 400 ms at 83.45% mean OOF balanced accuracy,
compared with 82.93% for the frozen 500 ms baseline. The directory contains
fold, seed, and aggregate metrics; OOF predictions; exact bin-equivalence
checks; a JSON protocol record; and a summary heatmap.

The completed Phase 2c horizon and rolling-streaming evidence is archived at:

```text
history/finger_movements/results/phase2c_horizon_diagnostic/
history/finger_movements/results/phase2c_streaming_causal/
```

They include fold metrics, seed-level OOF metrics, all per-case predictions at
ten 50 ms horizons, the accuracy-versus-latency summary, explicit
future-replacement invariance checks, a JSON protocol record, and a summary
figure. Official TEST was refused.

The corresponding completed runners are archived under
`history/finger_movements/experiments/`. The active causal checkpoint and fit
verification are under `models/finger_movements/cssd_lda/checkpoints/`; the
former zero-phase Phase 2b checkpoint is archived with its model.
