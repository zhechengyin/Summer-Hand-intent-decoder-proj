# Phase 4a Architecture Sweep

This directory contains the Optuna architecture-only search. Optimization settings, session-balanced sampling, feature construction, and month folds were frozen.

- 20 trials completed and 10 were pruned; 18 unique architectures were evaluated.
- The existing 64/64 model had selection score 0.4993 and 78,786 parameters.
- A 48/48, four-block, kernel-3, one-layer-GRU candidate scored 0.5038 with 45,266 parameters.

The sweep nominated 48/48 for multi-seed confirmation; it did not by itself authorize checkpoint replacement. `.cache/` contains expensive fold tensors retained for reproducibility.
