# CubeAI next-phase handoff

This phase stops at validated PyTorch checkpoints. Do not reuse archived ONNX,
H5, generated C, weights binaries, `.aibundle`, or `.memlib` artifacts with the
new weights.

## Inputs that are ready

- Shared architecture: `midsize/model.py`
- Per-fold metadata and SHA-256: each session's `manifest.json`
- Neural weights: five `fold-*.pt` files in every Midsize and Large session
  folder
- Preprocessing constants inside every checkpoint: selected channels, feature
  mean/std/floor, target mean/std, and seven-minute deployment policy

Midsize and Large contain byte-identical neural checkpoints for the same
session/fold. Convert the neural graph once per unique checkpoint; Large adds a
separate external-memory artifact after the base neural path is validated.

## Required next-phase sequence

1. Export each unique fold checkpoint to the existing split CubeAI graph
   contract, preserving the encoder and GRU/head interfaces.
2. Verify PyTorch ↔ exported model parity before CubeAI.
3. Run X-CUBE-AI analyze, validate, and generate for all 30 unique checkpoints.
4. Run generated-C parity with the same post-seven-minute held-out bins.
5. Package fold-specific constants/weights without choosing a test fold for the
   paper result.
6. For Large, expose GRU hidden state at timestep 49, rebuild one train-only
   memory bank per fold, tune retrieval on validation reaches only, and score
   held-out test reaches once.
7. Export the validated MCU memory format and run board latency/recall/R² tests.

## Completion gate

CubeAI is complete only when every manifest records generated artifact hashes,
PyTorch/exported/C parity tolerances, CubeAI version, graph ABI, and per-fold
status. Large is complete only when all 30 compatible memory banks exist and
the aggregate corrected R² is computed from all 30 test folds.
