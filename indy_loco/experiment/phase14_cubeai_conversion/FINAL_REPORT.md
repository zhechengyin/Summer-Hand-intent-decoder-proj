# Phase 14 final conversion report

All six per-session best-test-fold checkpoints were converted with X-CUBE-AI
10.2 and passed generated-C host replay. Midsize and Large use the same neural
weights, so each converted neural package is stored once and referenced by
both tiers.

| Session | Fold | FP32 R² | CubeAI C-encoder R² | R² drop | Gate |
|---|---:|---:|---:|---:|---|
| `indy_20160622_01` | 5 | 0.8634 | 0.8628 | +0.0006 | pass |
| `indy_20160630_01` | 4 | 0.7358 | 0.7363 | −0.0005 | pass |
| `indy_20170131_02` | 4 | 0.8503 | 0.8501 | +0.0001 | pass |
| `loco_20170210_03` | 5 | 0.7878 | 0.7871 | +0.0007 | pass |
| `loco_20170215_02` | 5 | 0.7269 | 0.7258 | +0.0011 | pass |
| `loco_20170301_05` | 1 | 0.8025 | 0.8026 | −0.0001 | pass |
| **Six selected folds** | — | **0.7944** | **0.7941** | **+0.0003** | **6/6 pass** |

The `0.7941` value is a deployment conversion diagnostic only. These six
folds were marked after observing test R², so this mean must not be reported as
the model's cross-validation performance.

For the full held-out replay, “CubeAI C-encoder R²” means the generated-C INT8
encoder followed by the FP32 GRU/head. The generated CubeAI GRU/head itself was
host-validated on held-out vectors with maximum absolute error below
`7.2e-7`, including the full state sequence and hidden[49].

The official paper-facing Midsize result is the average over every
validation-selected fold:

\[
R^2 = \mathbf{0.7411 \pm 0.0656}\quad (30\ folds).
\]

The canonical conversion ABI is `tcn64i8-gruseq64-xcai10-v1`: INT8 TCN
encoder with float32 I/O, followed by FP32 GRU/head with velocity and full
GRU-state-sequence outputs. Large can read `state[49]` as its 64-value
external-memory query. Compatible Phase-14 memlibs have not yet been built.

For Midsize, the same verified weights were regenerated as a velocity-only GRU
and promoted to firmware/GUI under `tcn64i8x6-gru64f32-p13-v3`. This keeps the
existing fast memory map and avoids an unused 12,800-byte hidden output. The
six GUI bundles, Phase-13 held-out masks, seven-minute replay assets, and mock
predictions were rebuilt. Physical board validation is the remaining gate.
