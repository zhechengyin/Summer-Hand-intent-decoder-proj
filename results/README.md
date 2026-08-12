# Results

The completed Phase 2e result is stored at:

```text
results/finger_movements/phase2e_lightweight_comparison/
```

ToeplitzLDA had the highest mean OOF BA at 84.50%, compared with 83.99% for
the baseline, but the gain was not consistent across seeds and did not improve
worst-seed BA or fold variability. The conditional nested fusion reached
84.09% and was also unstable. No model was promoted.

Completed FingerMovements evidence is archived under:

```text
history/finger_movements/results/phase2c_horizon_diagnostic/
history/finger_movements/results/phase2c_streaming_causal/
history/finger_movements/results/phase2c_bin_window_sweep/
history/finger_movements/results/phase2d_official_test_400ms/
```

The selected 400 ms window reached 83.99% mean OOF balanced accuracy, compared
with 82.93% for the 500 ms causal baseline. The active 400 ms checkpoint and
fit verification are under `models/finger_movements/cssd_lda/checkpoints/`.

The archived Phase 2d evaluation reports 77.00% accuracy and 77.05% balanced
accuracy on 100 official TEST cases. It is retrospective evidence, not a new
selection gate. The former 500 ms causal and zero-phase Phase 2b checkpoints
are archived with their model implementations under
`history/finger_movements/models/`.
