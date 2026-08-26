# Loco preparation

`prepare_loco_neurobench.py` converts the original Loco MAT sessions into a
lossless 4 ms representation compatible with NeuroBench's session-local
[primate-reaching protocol](https://github.com/NeuroBench/neurobench/blob/main/neurobench/datasets/primate_reaching.py).

Download or resume all ten published sessions with independent HTTP ranges and
verify their published byte sizes and MD5 checksums:

```bash
python indy_loco/data/processing/indy_loco/loco/download_loco_raw.py
```

Run all ten sessions from the repository root:

```bash
python indy_loco/data/processing/indy_loco/loco/prepare_loco_neurobench.py
```

Use `--benchmark-only` to process only the three official NeuroBench sessions.
Repeat `--session SESSION_NAME` to process explicitly selected sessions.
Use `--validate-only` to verify published raw checksums and any existing NPZ
artifacts without changing files.

The converter reads raw MAT files but never changes them. It preserves binary
spike presence at 4 ms, kinematics, channel order, reach boundaries, and the
official ordered 50/25/25 reach split. Model-specific windows and normalization
remain the responsibility of the Phase 7 experiment and must be fit from that
session's training reaches only.
