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

## Frozen sampling decision

The completed CPU seed 42/43/44 aggregate freezes `session` /
Session-balanced sampling for subsequent model development:

| Sampling | Validation loss, mean +/- SD | Validation R², mean +/- SD | Seed wins |
| --- | ---: | ---: | ---: |
| Window-weighted | 0.5329 +/- 0.0112 | 0.5080 +/- 0.0113 | 0/3 |
| **Session-balanced** | **0.5074 +/- 0.0221** | **0.5342 +/- 0.0198** | **3/3** |
| Month-balanced | 0.5259 +/- 0.0066 | 0.5183 +/- 0.0077 | 0/3 |

Session-balanced also has the highest session-macro validation R² (0.5468) and
the smallest train-validation R² gap (0.2321). It is now fixed for Optuna and
other hyperparameter comparisons; the sampling script remains only as a
reproducible experiment. January test data is still locked.

The completed baseline and three-sampler entry points now live under the root
[`history/`](../../history/README.md) folder. Their result JSON files remain
versioned as decision evidence.

## Completed Phase-1 Optuna sweep

`sweep_phase1_optuna.py` completed the seed-42 Phase-1 model-selection study. It is
self-contained: it does not import `common.py`, archived training scripts, or
code under `history/`. It jointly tunes:

- learning rate: `5e-5` to `1e-3`, log scale;
- AdamW weight decay: `1e-6` to `3e-2`, log scale;
- pre-GRU dropout: `0.10` to `0.50`, step `0.05`.

Every trial uses 20 epochs by default, seed 42, session-balanced training,
batch size 32, cosine decay, gradient clip 1, no augmentation, and identical
initial weights and sampled training windows. The current baseline
`lr=3e-4`, `weight_decay=1e-3`, `dropout=0.30` is queued as trial 0.

Only the 29 training and four December validation sessions are loaded. Pooled
validation normalized MSE is the Optuna objective; session-macro and
worst-session R² are stored as guardrails. The January test split is never
loaded.

Install the updated requirements and run from the repository root:

```bash
python -m pip install -r requirements.txt
python models/indy_32ch/sweep_phase1_optuna.py
```

The completed study contains 40 trials: 29 complete, 11 pruned, and zero failed.
Trial 32 is the objective winner (`lr=0.000913763`, `weight_decay=0.0137117`,
`dropout=0.10`, selected epoch 7), with validation loss 0.482066 and validation
R² 0.557579. Relative to the queued baseline, it reduced validation loss by
9.21% and improved the difficult December 6 session R² from 0.198875 to
0.319848.

For a short pipeline check:

```bash
python models/indy_32ch/sweep_phase1_optuna.py --validate-only
```

Do not resume the completed database with modified ranges. Its outputs are:

- `results/large/indy_32ch_phase1_optuna.db` — resumable Optuna study;
- `results/metrics/indy_32ch_phase1_optuna.json` — inspectable trial histories,
  selected metrics, protocol, and ranking;
- `results/figures/indy_32ch_phase1_optuna.png` — objective and parameter plots;
- `results/large/indy_32ch_phase1_best_checkpoint.pt` — current best
  validation-selected checkpoint.

Do not use the best Phase-1 checkpoint as the final model. Every top-five trial
used dropout 0.10, the lower search boundary, and the top two losses differ by
only 0.074%. First run a small separate boundary-refinement study including
dropout below 0.10, then confirm a few distinct finalists on seeds 42/43/44.
Later phases may examine paired-electrode dropout and model capacity. January
remains locked until every phase and the final seed protocol are frozen.
