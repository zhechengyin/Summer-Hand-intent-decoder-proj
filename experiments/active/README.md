# Active Experiments

## Phase 2c: causal bin/window sweep

`phase2c_bin_window_sweep.py` compares past-context windows of 200, 300, 400,
and 500 ms. Every window ends at the current prediction point A and each model
is refitted inside every TRAIN-only fold.

It also checks 10, 20, 50, and 100 ms streaming bins. Bin size changes how the
causal stream is divided into firmware updates; it does not change the samples
available at the endpoint. The script therefore verifies exact chunking
equivalence and reports identical accuracy for bins sharing the same window.

Run the complete sweep:

```bash
python experiments/active/phase2c_bin_window_sweep.py
```

Run one fold for every window plus all bin checks without writing files:

```bash
python experiments/active/phase2c_bin_window_sweep.py --validate-only
```

Default sweep grid:

```text
bin_ms:    10, 20, 50, 100
window_ms: 200, 300, 400, 500
seeds:     42, 43, 44
folds:     5
```

Complete TRAIN-only result:

| Window | Mean OOF BA | Seed SD | Worst seed |
|---:|---:|---:|---:|
| 200 ms | 80.17% | 1.56 pp | 78.17% |
| 300 ms | 82.81% | 1.30 pp | 81.33% |
| **400 ms** | **83.45%** | **1.07 pp** | **81.98%** |
| 500 ms | 82.93% | 1.03 pp | 81.67% |

The 400 ms window is provisional because its +0.52-point mean improvement did
not occur in all three seeds. The active 500 ms / 50 ms checkpoint has not
been overwritten. All bin-equivalence checks had exactly zero signal error.

The official dataset provides isolated 500 ms epochs rather than continuous
EEG. It can validate the endpoint model and chunked implementation, but cannot
validate filter-state behavior across successive real-world rolling windows.
That requires continuous recordings.

The completed Phase 2c horizon and rolling-streaming runners/results are
archived under `history/finger_movements/`. The active checkpoint is maintained
separately under `models/finger_movements/cssd_lda/`.
