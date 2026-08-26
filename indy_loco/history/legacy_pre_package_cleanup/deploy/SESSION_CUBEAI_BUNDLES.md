# Session Cube.AI weight bundles

The six session directories below `models/midsize` contain dynamically
loadable Cube.AI weight bundles. The promoted firmware path uses the validated
`cubeai_int8` artifacts:

1. six session-specific INT8 encoder graphs with a float32 external ABI,
   `(1, 192, 50)` to `(1, 50, 64)`
2. `indy_gru_head`: float32 `(1, 50, 64)` to `(1, 50, 2)`

The encoder graph must be selected by session because X-CUBE-AI embeds the
session-specific INT8 scales and zero-points in generated C. The uploaded
bundle still carries that session's encoder and GRU/head weights.

The component `.bin` files are emitted directly by X-CUBE-AI 10.2
`stedgeai generate --binary`. The project scripts do not reconstruct Cube.AI's
private weight layout.

## Reproduce and verify

Install the exact versions in `requirements-deploy.txt` in a Python 3.12
environment, then run from the repository root:

```powershell
python indy_loco/deploy/build_session_cubeai_int8.py
python indy_loco/deploy/promote_cubeai_int8.py
```

`build_session_cubeai_int8.py` performs post-training quantization using only
the selected fold's training reaches. It checks the full held-out fold against
the float32 checkpoint using the generated Cube.AI host library and rejects a
session when mean R2 drops by more than 0.01. It then emits the six graph source
sets and bundles. `promote_cubeai_int8.py` copies those checked artifacts into
the firmware and GUI repositories.

The original float32 builder and verifier remain available for regression and
fallback builds.

## Test from a clean checkout

The promoted firmware graph sources and the GUI's six `.aibundle` files are
checked-in artifacts. A second computer does not need the training datasets,
PyTorch, TensorFlow, or X-CUBE-AI to run Dataset Replay:

1. build and flash the CM7 **Release** configuration;
2. install `BCI-STM32-Plot/requirements.txt` and launch the GUI;
3. select one of the six Dataset Replay sessions and start it.

The training/export environment and X-CUBE-AI 10.2 CLI are needed only when
regenerating or auditing the INT8 artifacts.

## Graph ABI contract

The firmware must only load these weights into generated graph code carrying
this exact identifier:

```text
tcn64i8x6-gru64f32-xcai10-v2
```

This identifier covers the split, tensor shapes, INT8 encoder / FP32 GRU
representation, six graph variants, X-CUBE-AI 10.2 conversion family and the
legacy C API used by the generated graphs. It is separate from the container
version. A firmware graph or conversion-layout change requires a new ABI
identifier even when component sizes happen to remain unchanged.

## Bundle v1 binary layout

All integer fields are little-endian. Component offsets are 32-byte aligned.
The fixed header is 256 bytes.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | magic `BCIAIB1\0` |
| 8 | 2 | bundle format version, currently 1 |
| 10 | 2 | header size, 256 |
| 12 | 4 | flags; bit 0 indicates a parameter block |
| 16 | 4 | total bundle size |
| 20 | 32 | NUL-padded model/session ID |
| 52 | 2 | source channel count, 96 for Indy or 192 for Loco |
| 54 | 2 | selected channel count, 96 |
| 56 | 2 | feature count, 192 |
| 58 | 2 | window bins, 50 |
| 60 | 2 | output timestep, 49 |
| 62 | 2 | required alignment, 32 |
| 64 | 32 | SHA256 of `deployment_candidate.pt` |
| 96 | 12 | encoder offset, size and CRC32 |
| 108 | 12 | GRU/head offset, size and CRC32 |
| 120 | 12 | parameter block offset, size and CRC32 |
| 132 | 32 | SHA256 of the full bundle body after the header |
| 164 | 32 | container ID `bci-cubeai-bundle-v1` |
| 196 | 4 | header CRC32, computed with this field zeroed |
| 200 | 32 | NUL-padded graph ABI ID |
| 232 | 24 | reserved, zero |

The 976-byte parameter block contains:

| Offset | Type | Count | Content |
|---:|---|---:|---|
| 0 | float32 | 192 | `feature_std_floor` |
| 768 | float32 | 2 | `target_mean` |
| 776 | float32 | 2 | `target_std` |
| 784 | uint16 | 96 | selected source-channel indices |

The Loco mappings therefore preserve their per-session 192-to-96 channel
selection without changing the common 192-feature model interface.

## Runtime acceptance order

Before graph initialization, firmware should validate magic, format version,
total size, header CRC32, expected graph ABI ID, model ID, aligned/nonoverlapping
component ranges, per-component CRC32, parameter CRC32 and body SHA256. It
should switch the active weights only after all checks succeed.
