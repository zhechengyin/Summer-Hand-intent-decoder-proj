# Active Indy Loco experiment

## Phase 6: 96-channel training

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
model checkpoints are never modified automatically.
