# CubeAI and Large-memory handoff

The six highlighted per-session best folds are converted and host-validated.
Midsize and Large share each neural package, so there are six unique CubeAI
packages rather than twelve copies.

## Ready now

- Six INT8-encoder + FP32-GRU/head CubeAI packages under
  `midsize/<session>/cubeai/fold-<best>/`.
- Six Large reference manifests under `large/<session>/cubeai/manifest.json`.
- Full GRU state output `[1, 50, 64]`; hidden[49] is row 49 and can seed the
  new residual-memory query.
- Generated-C held-out replay passed all six accuracy gates. The selected-fold
  diagnostic mean is 0.7941, while official reporting remains **0.7411 ±
  0.0656 across all 30 folds**.

## Remaining sequence

1. Build a compatible train-only GRU-hidden[49] memlib for the first Indy
   package and validate retrieval on PC.
2. Integrate that one neural package and memlib into firmware/GUI and run board
   latency, parity, bank-ABSENT, and bank-READY tests.
3. If the pilot board result passes, build the remaining five best-fold
   memlibs and integrate all six session packages.
4. To obtain a paper-facing Large result, separately build/evaluate all 30
   fold-specific banks with validation-only tuning. Never average only the six
   test-selected folds for the paper.

CubeAI 10.2 cannot import GRU `return_state`, and its Keras importer fails on
`Cropping1D`. The active ABI therefore exposes the complete GRU state sequence
and reads timestep 49 without recomputing the GRU. This adds a 12.8 KB float32
output view/buffer requirement that must be included in the MCU RAM audit.
