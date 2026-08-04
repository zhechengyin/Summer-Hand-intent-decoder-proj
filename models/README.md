# Models

There is one active model:

```text
models/finger_movements/feature_logistic/
```

It contains the Feature + Logistic definition and preprocessing contract. The
representation is retained from Phase 1c; Phase 1d replaced AdamW + dropout
training with L2 Logistic Regression. `C=1` is the current candidate and still
requires nested-CV confirmation, so no final-training entry point or checkpoint
is active yet. See its [README](finger_movements/feature_logistic/README.md).

The retired AdamW implementation is archived under
`history/finger_movements/models/feature_linear_adamw/`.
