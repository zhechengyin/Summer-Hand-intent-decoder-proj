# Results

There is no active result set. Completed FingerMovements evidence is archived
under `history/results/`, including:

```text
history/results/phase2c_horizon_diagnostic/
history/results/phase2c_streaming_causal/
history/results/phase2c_bin_window_sweep/
history/results/phase2d_official_test_400ms/
history/results/phase2e_lightweight_comparison/
history/results/phase2f_riemannian/
```

The selected 400 ms window reached 83.99% mean OOF balanced accuracy, compared
with 82.93% for the 500 ms causal baseline. The active 400 ms checkpoint and
fit verification are under `models/cssd_lda/checkpoints/`.

The archived Phase 2d evaluation reports 77.00% accuracy and 77.05% balanced
accuracy on 100 official TEST cases. It is retrospective evidence, not a new
selection gate. The former 500 ms causal and zero-phase Phase 2b checkpoints
are archived with their model implementations under
`history/models/`.

Phase 2f reached 85.13% mean OOF balanced accuracy but increased seed/fold
variability and failed the frozen promotion rule. It did not create or modify
a checkpoint. The active Phase 2c checkpoint remains under
`models/cssd_lda/checkpoints/`.
