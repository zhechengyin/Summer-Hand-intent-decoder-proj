# Active Indy Loco experiment

## Phase 6: completed channel and regularization diagnosis

The sweep is complete. The winner is the all-96, kernel-3, four-block 64/64
TCN+GRU with 0.20 paired channel dropout. Across seeds 42–44 it reached pooled
December validation R² `0.7004 ± 0.0019` and macro R² `0.7023 ± 0.0015`.
The seed-43 epoch-15 checkpoint was promoted to
`../../models/indy_96ch/phase6_96ch_64x64_checkpoint.pt`. January was never
loaded.

The initial `phase6_96channel_training.py` run is complete. Its seed-43 CUDA
checkpoint selected epoch 5: train R² `0.8283`, pooled December validation R²
`0.6439`, macro R² `0.6466`, and worst-session R² `0.5444`. Continued epochs
raised train R² but not validation R², motivating the controlled sweep below.

`phase6_channel_structure_regularization_sweep.py` keeps the 64/64 width and
Phase 5 optimization settings fixed. Seed 43 screens:

- activity Top-64;
- stability-aware Top-64/72/80/88;
- all 96 channels;
- all-96 kernel size 2;
- all-96 three-block TCN;
- all-96 paired channel dropout 0.10 and 0.20.

The stability score uses only training-session 60-second prefixes: activity,
cross-session variability, silent-session frequency, and drift across training
months. December never contributes to channel ranking. After screening, only
the channel winner, all-96 structure/regularization winner, activity Top-64
reference, and all-96 baseline are confirmed with seeds 42 and 44. Depending on
duplicate winners, the script performs at most 18 fits.

Validate without loading arrays:

```bash
python indy_loco/experiments/active/phase6_channel_structure_regularization_sweep.py \
  --validate-only
```

Run the sweep:

```bash
python indy_loco/experiments/active/phase6_channel_structure_regularization_sweep.py \
  --threads 4
```

Resume an interrupted run:

```bash
python indy_loco/experiments/active/phase6_channel_structure_regularization_sweep.py \
  --threads 4 --resume
```

CUDA is selected automatically when available; otherwise the runner uses CPU.
The state signature records the selected backend and prevents a resumed result
from mixing CPU and CUDA fits. January is never loaded. The runner's result
checkpoints remain immutable experiment evidence; promotion is a separate,
explicit copy into `models/`.

## Initial Phase 6 runner

`phase6_96channel_training.py` performs one controlled 20-epoch run of the
strictly causal 64/64 TCN+GRU using all 96 physical neural channels:

- input: 96 raw 40 ms count streams plus 96 causal-EWMA streams;
- history: 50 past bins (2 seconds), with no future samples;
- calibration: first 60 seconds of each session;
- training: all 29 chronological train sessions, session-balanced;
- optimization: seed 43, LR 0.0009, WD 0.025, dropout 0.10;
- validation: four December sessions, inference/checkpoint selection only;
- test: January is registered but never loaded;
- device: NVIDIA CUDA when available, otherwise CPU; Apple MPS is disabled.

Phase 5 showed that detector-based removal of the June 30 and October 13
sessions did not improve mean validation R². Phase 6 therefore keeps all 29
training sessions. The detector remains a runtime safety mechanism, not a
default data-cleaning rule.

From the repository root, first validate the protocol:

```bash
python indy_loco/experiments/active/phase6_96channel_training.py \
  --validate-only
```

Run the training:

```bash
python indy_loco/experiments/active/phase6_96channel_training.py \
  --threads 4
```

The default `--device auto` works on both machines: it selects CUDA on a Windows
computer with a CUDA-enabled PyTorch installation and falls back to CPU on the
Mac. To force a device, use `--device cuda` or `--device cpu`.

Outputs are written under `../../results/phase6_96channel_training/`. The best
epoch is selected by pooled December validation loss. Existing Phase 6 outputs
are protected; `--overwrite` is required for an intentional rerun. Retained
model checkpoints are never modified automatically. This runner is retained to
reproduce the initial all-96 baseline; new diagnosis should use the sweep above.
