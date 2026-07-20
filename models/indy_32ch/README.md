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

## Active Phase-1 Optuna sweep

`sweep_phase1_optuna.py` is the only active model-selection entry point. It is
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

The default adds 40 trials. For a short pipeline check:

```bash
python models/indy_32ch/sweep_phase1_optuna.py --validate-only
```

The SQLite study is resumable: running the default command again adds another
40 trials to the same study. Completed and pruned trials are preserved if the
process is interrupted. Outputs are:

- `results/large/indy_32ch_phase1_optuna.db` — resumable Optuna study;
- `results/metrics/indy_32ch_phase1_optuna.json` — inspectable trial histories,
  selected metrics, protocol, and ranking;
- `results/figures/indy_32ch_phase1_optuna.png` — objective and parameter plots;
- `results/large/indy_32ch_phase1_best_checkpoint.pt` — current best
  validation-selected checkpoint.

Do not use the best Phase-1 checkpoint as the final model. The top five trials
must first be confirmed on seeds 43 and 44, after which later phases may examine
paired-electrode dropout and model capacity. January remains locked until every
phase and the final seed protocol are frozen.
