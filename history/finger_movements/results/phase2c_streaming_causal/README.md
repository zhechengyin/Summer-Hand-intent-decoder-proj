# Phase 2c Past-Only Streaming Validation

This experiment verified the causal 500 ms endpoint model with stateful 50 ms updates.

- Mean OOF BA: 82.93%.
- Seed SD: 1.03 pp; worst seed: 81.67%.
- Ten 50 ms filter calls reproduced one full causal pass.
- After startup, adding one new 50 ms bin produced the next output.
- Extreme samples appended after the current point changed current features and predictions by exactly zero.

The result validated implementation semantics on isolated epochs. It did not validate long-running state on continuous EEG. The later window sweep selected 400 ms instead of 500 ms.
