# Phase 12 evidence: GRU hidden[49] versus Encoder[49]

## Claim supported by the saved experiment

Under the controlled Phase-12 protocol, the pure `gru_hidden_49` query has a
higher held-out corrected mean R² than the pure `encoder_49` query in all six
benchmark sessions.

| Session | Encoder corrected R² | GRU corrected R² | GRU − Encoder |
|---|---:|---:|---:|
| indy_20160622_01 | 0.81034 | 0.81526 | +0.00492 |
| indy_20160630_01 | 0.66955 | 0.67842 | +0.00886 |
| indy_20170131_02 | 0.69087 | 0.71490 | +0.02403 |
| loco_20170210_03 | 0.64628 | 0.65404 | +0.00776 |
| loco_20170215_02 | 0.56132 | 0.56725 | +0.00593 |
| loco_20170301_05 | 0.73743 | 0.73758 | +0.00015 |

The unweighted session mean difference is **+0.00861 R²**. A 100,000-repeat
session bootstrap gives a 95% interval of **[+0.00366, +0.01530]**. The exact
one-sided paired Wilcoxon and exact one-sided sign tests both give
**p = 0.015625**; the two-sided Wilcoxon sensitivity result is **p = 0.03125**.

## Protocol that must match before comparing another result

- Saved per-session **fold-1** Midsize checkpoint, frozen during the ablation.
- Identical train/validation/test reaches for every representation.
- Bank residuals and both PCA fits use train reaches only.
- Retrieval K, temperature, and blend are selected on validation reaches only.
- Final corrected R² is evaluated once on held-out test reaches.
- Query representation: timestep 49 of the same 50-bin causal model input.
- Representation PCA: 32 dimensions.
- Long-context PCA: 32 dimensions, weight 0.5.
- Final normalized key: 64 dimensions.
- Exact PC cKDTree retrieval; maximum 128 neighbours.
- Per-session uncertainty resamples reaches, not autocorrelated 40 ms bins.
- Cross-session inference treats the six sessions as the independent units.

The exact constants and search grid are saved in `parameters.json` and the
complete results are in `results/cross_session/` and
`results/by_session/<session>/metrics.json`.

## What this evidence does not establish

This result contradicts a blanket claim that Encoder[49] is generally better
than GRU hidden[49] under the same protocol. It does not invalidate a result
from a different fold, checkpoint, preprocessing policy, context definition,
PCA budget, retrieval grid, split, or aggregation unit. A conflicting test
must publish those choices before the results are comparable.

Two session-level reach-bootstrap intervals cross zero, so the within-session
advantage is not individually significant everywhere. Also, among all four
Phase-12 candidates, `encoder_gru_49` wins validation in two sessions; the
specific supported comparison here is pure GRU versus pure Encoder.

## Reproduce and validate

```bash
.venv-deploy/bin/python \
  indy_loco/experiments/phase12_external_memory_representation_ablation/run_all_sessions.py \
  --device cpu --threads 4 --overwrite

.venv-deploy/bin/python \
  indy_loco/experiments/phase12_external_memory_representation_ablation/aggregate_sessions.py

.venv-deploy/bin/python \
  indy_loco/experiments/phase12_external_memory_representation_ablation/validate_artifacts.py
```

The included `.memlib` files are self-describing PC experiment archives. They
are not claimed to be drop-in BCIMEM firmware binaries.
