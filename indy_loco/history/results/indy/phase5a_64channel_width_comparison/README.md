# Phase 5a 64-Channel Width Comparison

This is an exploratory, single-seed comparison using 64 neural channels and 64 causal EWMA features. The authoritative run used CPU, seed 43, 30 epochs, train-only fitting, and December validation selection. January was not loaded.

| Architecture | Parameters | Selected epoch | Validation loss | Pooled R² | Macro R² | Worst R² |
|---|---:|---:|---:|---:|---:|---:|
| 64/64 | 82,882 | 7 | 0.3571 | 0.6625 | 0.6669 | 0.5842 |
| 48/48 | 48,338 | 4 | 0.3613 | 0.6569 | 0.6608 | 0.5760 |

The 64/64 model ranked first. It was not promoted because the comparison lacked multi-seed confirmation and the existing detector was calibrated to a different 32-channel input. An earlier unstable MPS run and the stored 32/32 artifact are withdrawn and should not be cited as valid comparisons.
