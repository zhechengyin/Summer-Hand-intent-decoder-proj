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

## Eight-session causal smoke test

`train_smoke_test.py` trains the candidate on the eight locally available
sessions using the fixed split `train1..train6 / eval1 / test1`. It prints
train/eval/test normalized MSE and test R² every epoch, selects its diagnostic
checkpoint on eval R² only, and writes metrics plus two training-curve figures.

```bash
../venv/bin/python models/indy_32ch/train_smoke_test.py
```

This is not promotion evidence: `test1` is historically reused, and the script
does not replace nested leave-one-month-out validation.

Latest run (2026-07-17, seed 42, 60 epochs): 78,786 parameters and
1,354/217/206 train/eval/test windows. Eval R² selected epoch 32; that checkpoint
scored eval R² 0.5781 and diagnostic test R² 0.5851. See
`results/metrics/indy_32ch_smoke_test.json` for the full per-epoch history.
