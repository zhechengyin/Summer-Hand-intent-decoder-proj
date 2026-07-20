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

## Chronological causal baseline

`train_chronological_baseline.py` uses all 37 processed Indy sessions with the
fixed chronological split: 29 train, 4 validation, and 4 test. The default run
is 25 epochs. Only training sessions can update model weights, select the 32
channels, or fit target-normalization statistics. Validation and test are
inference-only diagnostics and never select a checkpoint. The checkpoint is
selected using training loss only.

The baseline uses a conservative learning rate, no input augmentation by
default, gradient clipping, and a training-derived standard-deviation floor.
The floor prevents a channel that is silent during the causal 60-second
calibration from turning a later spike into an extremely large normalized value.

Every epoch prints train/validation/test normalized MSE and mean velocity R2.
The run writes a checkpoint, a complete JSON history, and one figure containing
the three loss curves and three R2 curves.

```bash
../venv/bin/python models/indy_32ch/train_chronological_baseline.py
```

The four test sessions are displayed every epoch only for the requested
diagnostic curve. Their values must not be used to change the architecture,
hyperparameters, epoch count, preprocessing, or checkpoint. For unbiased model
selection, use validation or nested leave-one-month-out evaluation and inspect
the locked test result only after the full configuration is frozen.

## Sampling comparison

`train_sampling_comparison.py` runs a multi-seed comparison of three otherwise
identical sampling policies. By default it imports the completed compatible
CPU seed-42 result and trains seeds 43 and 44, producing six new arms and a
three-seed/nine-arm aggregate:

1. `window` — every training window appears once per epoch, so long sessions
   naturally have more weight;
2. `session` — all 29 training sessions contribute equally per epoch;
3. `month` — the four training months contribute equally, then sessions are
   balanced within each month.

Within each newly trained seed, every strategy uses the same initial weights,
preprocessing, optimizer, number of samples per epoch, and 25 epochs.
Validation remains inference-only. Checkpoints use minimum pooled validation
loss, exactly matching the existing seed-42 run; session-macro loss and
worst-session R² are additional diagnostics. The script rejects reuse if the
requested configuration or chronological split differs from seed 42. The
January test artifacts are not loaded. CPU is the default because the current
MPS path failed the CPU/MPS parity diagnostic.

Run the complete comparison from the repository root:

```bash
python models/indy_32ch/train_sampling_comparison.py
```

The terminal reports each seed/strategy/epoch, pooled and session-macro
validation metrics, worst-session R², checkpoint selection score, gradient
norms, and month exposure. The final section reports mean ± sample SD across
all three seeds and the number of per-seed wins. It writes six new seed-specific
checkpoints, a complete JSON result at
`results/metrics/indy_32ch_sampling_seed_sweep.json`, and a mean ± SD figure at
`results/figures/indy_32ch_sampling_seed_sweep.png`. The earlier seed-42 files
are deliberately preserved as historical evidence.

### Frozen sampling decision

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

For a short pipeline check without committing to the full comparison:

```bash
python models/indy_32ch/train_sampling_comparison.py \
  --seeds 43 \
  --no-reuse-seed42 \
  --epochs 1 \
  --device cpu
```
