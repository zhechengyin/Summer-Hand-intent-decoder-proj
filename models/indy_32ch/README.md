# Indy 32-channel candidate

This directory intentionally contains no checkpoint yet. It is the destination
for the model promoted after the causal-prefix normalization and nested detector
evaluation in `docs/ROADMAP.md`.

Candidate configuration: `configs/indy_32ch.yaml`.

Required before promotion:

1. frozen session manifest and outer-fold protocol;
2. counts + causal EWMA input and causal target velocity;
3. 60-second past-only normalization warm-up with its outputs discarded;
4. trained checkpoint and normalization metadata;
5. int8 export with accuracy comparison;
6. measured STM32 flash, RAM, and inference time.
