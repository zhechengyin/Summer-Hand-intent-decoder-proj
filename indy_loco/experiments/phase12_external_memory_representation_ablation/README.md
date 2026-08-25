# Phase 12 — external-memory representation ablation

For the shortest evidence trail supporting the six-session GRU-versus-Encoder
claim, start with [`CLAIM_EVIDENCE.md`](CLAIM_EVIDENCE.md), the machine-readable
[`parameters.json`](parameters.json), and
[`results/cross_session/gru_vs_encoder_by_session.csv`](results/cross_session/gru_vs_encoder_by_session.csv).

This experiment freezes the saved fold-1 Midsize checkpoint for
`indy_20160622_01` and tests whether retrieval keys built from the GRU state,
TCN encoder state, their concatenation, or a masked 50-step encoder mean make
the memory residual more locally predictable.

The comparison is controlled: every representation is PCA-compressed to 32D,
combined with the same 32D long-context key, and searched against the same
train-only residual bank. Hyperparameters are chosen on validation reaches;
the held-out test reaches are used only for the final table. Reach bootstrap
confidence intervals account for within-reach temporal dependence.

Run with the repository deployment environment:

```bash
.venv-deploy/bin/python \
  indy_loco/experiments/phase12_external_memory_representation_ablation/run.py \
  --protocol-check-only

.venv-deploy/bin/python \
  indy_loco/experiments/phase12_external_memory_representation_ablation/run.py \
  --device cpu --threads 4
```

Run all benchmark sessions and aggregate the session-level inference:

```bash
.venv-deploy/bin/python \
  indy_loco/experiments/phase12_external_memory_representation_ablation/run_all_sessions.py \
  --device cpu --threads 4 --overwrite

.venv-deploy/bin/python \
  indy_loco/experiments/phase12_external_memory_representation_ablation/aggregate_sessions.py
```

Run the GUI-selected checkpoint with deployment-parity preprocessing, comparing
the same rolling replay with the GRU bank `ABSENT` and `READY`:

```bash
.venv-deploy/bin/python \
  indy_loco/experiments/phase12_external_memory_representation_ablation/deployment_parity_ab.py \
  --protocol-check-only --device cpu --threads 4

.venv-deploy/bin/python \
  indy_loco/experiments/phase12_external_memory_representation_ablation/deployment_parity_ab.py \
  --device cpu --threads 4 --overwrite

.venv-deploy/bin/python \
  indy_loco/experiments/phase12_external_memory_representation_ablation/validate_deployment_parity.py
```

Outputs are written only beneath this experiment directory, except for the
new memory-library artifacts, which are written to
the matching session directory beneath `indy_loco/models/large/` as requested.

The `*.memlib` outputs use the self-describing `phase12_pc_memlib_v1` NumPy
archive schema. They intentionally do **not** claim compatibility with the
current fixed-width `BCIMEM` firmware ABI; the GRU and concatenated variants
need the MCU model graph to expose their matching intermediate query.
