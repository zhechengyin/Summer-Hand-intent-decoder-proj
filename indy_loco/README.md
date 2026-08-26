# Indy/Loco decoder packages

The authoritative active surface contains exactly **12 session packages**:
six Midsize packages and six Large packages. Start from
[`models/manifest.json`](models/manifest.json); do not select a checkpoint from
`history/`.

## Canonical sessions and scores

Every package uses the checkpoint from the fold with the highest Phase-7 test
R² for that session. This is an explicit GUI/deployment demonstration policy,
not an unbiased generalization estimate.

| Session | Selected fold | Phase-7 chunked test R² | Midsize rolling R² | Large rolling R² |
|---|---:|---:|---:|---:|
| indy_20160622_01 | 5 | 0.8216 | 0.7311 | 0.8115 |
| indy_20160630_01 | 4 | 0.7004 | 0.5377 | 0.7016 |
| indy_20170131_02 | 4 | 0.7998 | 0.5660 | 0.7838 |
| loco_20170210_03 | 5 | 0.7067 | 0.6421 | 0.6982 |
| loco_20170215_02 | 4 | 0.6981 | 0.5133 | 0.6751 |
| loco_20170301_05 | 1 | 0.7500 | 0.7444 | 0.7672 |
| **Mean** | — | **0.7461** | **0.6224** | **0.7396** |

The Phase-7 number uses reach-local chunked preprocessing. The Midsize and
Large numbers use the same continuous rolling deployment preprocessing, so
only the final two columns form a matched deployment A/B.

## Package definitions

### Midsize

`models/midsize/<session>/` contains `checkpoint.pt` and `manifest.json`.
The shared 86,978-parameter causal TCN+GRU is in `models/midsize/model.py`.
`models/midsize/runtime.py` requires an explicit session and implements
60-second calibration followed by stride-1 rolling inference.

### Large

`models/large/<session>/` contains `checkpoint.pt`, `memory.memlib`, and
`manifest.json`. Large means **the same Midsize neural base plus GRU external
memory**; it is not a separately trained larger neural network.

The memory uses 32D GRU PCA + 32D long-context PCA, int8 keys and FP16
residuals. Reported retrieval used exact PC cKDTree search. The memlib is not
yet a firmware BCIMEM/IVF binary, so the Large score is a PC memory-quality
result rather than an STM32 latency result.

## Validation

```bash
.venv-deploy/bin/python indy_loco/models/package_tools.py validate
```

The validator requires exactly 12 packages, verifies SHA-256 hashes, checks
that both tiers share the same session checkpoint, and validates memory schema,
dimensions and dtypes.

## Archive boundary

Previous experiments, checkpoints, generated CubeAI material, results and old
descriptions are under `history/`. They are retained only for provenance and
are **not active model-selection guidance**. Local datasets remain under
`data/` and are not part of the package contract.
