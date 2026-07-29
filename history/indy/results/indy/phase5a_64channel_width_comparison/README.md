# Phase 5a 64-channel width comparison

Phase 5a compares two strictly causal TCN+GRU widths while expanding the neural
input from 32 to 64 selected channels:

- 64 TCN filters / 64 GRU hidden units;
- 48 TCN filters / 48 GRU hidden units.

Both candidates use 64 raw count streams plus 64 causal-EWMA streams, giving a
128-by-50 input window. Channel selection uses only the first 60 seconds of
the 29 train sessions. Both candidates use CPU, seed 43, session-balanced
sampling, the frozen optimizer settings and a complete 30-epoch cosine
schedule.

December validation is inference-only and selects the minimum-loss checkpoint.
January is never loaded. The two checkpoints remain experiment artifacts and
cannot replace the active 32-channel runtime: both detector layers must be
refitted for the new channel mapping, and Layer 2 must match the chosen GRU
hidden width.

Run from the repository root:

```bash
python experiments/active/phase5a_64channel_width_comparison.py --device cpu
```

Phase 5a is deliberately CPU-only. PyTorch 2.13.0 on Apple MPS reproduced
incorrect backward gradients for this graph even though forward predictions
matched CPU. `--validate-only` checks the protocol without loading arrays or
writing output.

If the CPU 64/64 checkpoint has already completed, train only the replacement
48/48 candidate and reuse the saved 64/64 evidence:

```bash
python experiments/active/phase5a_64channel_width_comparison.py \
  --device cpu \
  --architectures 64ch_48x48
```

The omitted 64/64 architecture is checksum-read from its existing Phase-5a
checkpoint and is not retrained or overwritten.

The first run on 2026-07-28 used the earlier unsafe `auto` default and selected
MPS. Its checkpoints, metrics, CSV and figure are invalid and must be replaced
with a CPU run using `--overwrite`.

An intermediate CPU run used an unintended 32/32 comparison. That architecture
is withdrawn from Phase 5a and its artifact is not eligible for selection.
The registered comparison is only 64/64 versus 48/48.
