# Processed Loco sessions

Each Loco session is stored as one compressed NPZ artifact. The representation
preserves the official NeuroBench session-local benchmark inputs at 4 ms and is
not normalized or pooled across sessions.

Important arrays:

| Array | Shape | Meaning |
|---|---|---|
| `spike_presence` | `(192, time)` | Binary per-channel spike presence at 4 ms after combining sorted units |
| `velocity_per_sample` | `(time, 2)` | Official cursor-position gradient target; displacement per 4 ms sample |
| `cursor_position` | `(time, 2)` | Raw cursor position |
| `target_position` | `(time, 2)` | Raw target position |
| `timestamps_s` | `(time,)` | Original timestamps in seconds |
| `channel_names` | `(192,)` | Physical channel order |
| `reach_bounds` | `(reaches, 2)` | Start-inclusive, end-exclusive reach boundaries |
| `reach_split` | `(reaches,)` | Official ordered split: `0=train`, `1=validation`, `2=test` |

The official benchmark uses three sessions (`20170210_03`, `20170215_02`, and
`20170301_05`) and trains a separate decoder per session. Phase 7 will construct
model windows and training-only normalization without crossing reach or split
boundaries. The remaining seven published sessions are supported for later
robustness analysis only when explicitly processed; they are never silently
mixed into the official comparison.

`velocity_per_sample` intentionally reproduces the official central-gradient
target for benchmark comparability. It is an offline label definition, not a
claim that future kinematics are available to firmware. Any deployment-focused
causal-target experiment must be reported separately from the paper benchmark.
