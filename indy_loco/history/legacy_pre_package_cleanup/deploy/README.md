# Indy Phase 6 / Phase 9 Cube.AI deployment bundle

This directory is a self-contained conversion and validation bundle for the
promoted 96-channel, 64/64 causal TCN+GRU checkpoint. No STM32 firmware files
are modified by this bundle.

## Deployable model

X-CUBE-AI 10.2 accepts the normal PyTorch ONNX `GRU` node, but its generated
host C implementation does not reproduce ONNX `linear_before_reset=1`.
Unrolling all 50 GRU steps is numerically correct but makes `stedgeai` exceed
the host memory limit. The verified deployment therefore uses two consecutive
Cube.AI networks:

1. `indy_encoder`: ONNX LayerNorm + causal TCN
   - input: float32 `(1, 192, 50)`
   - output: float32 `(1, 50, 64)`
2. `indy_gru_head`: Keras reset-after GRU + linear head
   - input: float32 `(1, 50, 64)`
   - output: float32 `(1, 50, 2)`

The Keras GRU weights are reordered from PyTorch `[reset, update, new]` to
Keras `[update, reset, new]`. The two-network output is equivalent to the
original checkpoint. Phase 9 reads output timestep 49.

## Verified results

| Item | Encoder | GRU + head | Combined |
|---|---:|---:|---:|
| Weights | 248,156 B | 100,360 B | 348,516 B |
| Activations | 76,800 B | 27,136 B | 103,936 B conservative total |
| MACC | 3,542,560 | 1,241,700 | 4,784,260 |

Generated-C chain versus the original PyTorch checkpoint, over eight Phase 9
and deterministic samples:

- RMSE: `1.3665e-7`
- maximum absolute error: `6.2585e-7`
- timestep-49 maximum absolute error: `1.4901e-7`
- cosine similarity reported by Cube.AI: `1.0`

The activation total assumes independent activation arrays for both networks.
Later firmware integration can evaluate a shared lifetime-aware memory pool.

## Important files

- `model/indy_phase6_encoder.onnx`: deployable encoder model.
- `model/indy_phase6_gru_head.h5`: deployable GRU/head model.
- `cubeai/encoder/generated/`: generated encoder C/H files.
- `cubeai/gru_head/generated/`: generated GRU/head C/H files.
- `cubeai/encoder/validate/`: encoder host-C validation.
- `cubeai/gru_head/validate/`: GRU/head host-C validation.
- `cubeai/end_to_end/`: generated-C chain validation.
- `cubeai/runtime/`: X-CUBE-AI headers and STM32H7 CM7 GCC runtime library.
- `validation/`: inputs, reference values and parity reports.
- `metadata/deployment_manifest.json`: hashes, interfaces and measured sizes.
- `source/`: copied checkpoint, model source and Phase 9 golden vectors.
- `diagnostics/`: failed single-network compatibility experiments retained for
  traceability; these are not the deployable artifacts.

## Reproduce

From `Summer-Hand-intent-decoder-proj`:

```bash
../venv/bin/python indy_loco/deploy/export_model.py
MPLCONFIGDIR=/private/tmp/matplotlib-cache \
  ../venv/bin/python indy_loco/deploy/export_split_models.py
indy_loco/deploy/run_cubeai.sh
```

`export_model.py` prepares the original PyTorch reference values and deployment
constants. `export_split_models.py` creates the two deployable model files.
`run_cubeai.sh` runs analyze, validate, generate and end-to-end host-C
validation with strict error propagation.

This validation is on the Cube.AI host C model. Target CM7 compilation, timing,
cache placement and firmware integration remain intentionally pending.
