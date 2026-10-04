# Phase20: larger model for demonstrating external SDRAM capability

Measured average AI latency is **35.8468 ms (approximately 35.9 ms)** on STM32H747 CM7 at 480 MHz. The validation-selected fallback is minGRU width 96/depth 3, with INT8 matrix kernels, exact offline GELU tables and FP32 normalization/recurrence. It preserves 96-channel counts, seven-minute calibration, independent 50-bin windows and zero recurrent state at each window start.

Full Indy replay delivered all 50,743 expected predictions and all 3,815 held-out windows, with zero input losses, coalesced predictions or AI overruns. AI p95 is 35.920 ms and maximum 35.991 ms. End-to-end p95 is 36.205 ms; rare queue delays produced an end-to-end maximum of 68.044 ms. The held-out board R² is 0.86866426 (vx 0.89369875, vy 0.84362984), versus matched Midsize 0.84651369. Board/host maximum prediction difference is 1.1920929e-7.

Useful SDRAM weights/constants total **112,128 bytes**. The remaining 117,012 weight bytes are in DTCM; D1 contains 105,600 activation bytes and 98,304 lookup bytes. Staging, inactive tables and unused reservations do not count toward useful SDRAM data.

## Accuracy and pause boundary

This is a saved, paused implementation, **not completed 30-fold qualification**. Nine selected-model checkpoints are trained; eight folds are evaluated. Their mean R² is 0.76317181 versus paired Midsize 0.75870344 (+0.00446837). The final six-session × five-fold gate remains incomplete. The winner was frozen using validation and board latency before its test evaluation; do not select another architecture using these partial test scores.

`release/` preserves the frozen selection, compact numerical receipts, selected checkpoint, static INT8 ONNX and full-replay evidence. Raw logs, intermediate Cube.AI workspaces, other checkpoints and bulk prediction arrays remain local under ignored `results/` directories. Nothing is scheduled to resume automatically.

## Reproduction

From the repository root use `.venv/Scripts/python.exe -X utf8 -m indy_loco.experiment.board_latency_20261004.<module>`:

- `train`: prescribed fold-1 screen only; seed 43 and frozen b2 recipe.
- `export <checkpoint>`: independent-window ONNX and train-only static quantization.
- `validate_generated <checkpoint> --work <Cube-workspace> --graphs <reference-and-fused-directory> --label lut`: fixed 256 probes and full validation at unchanged `atol=2e-4, rtol=1e-4`.
- `package_board <checkpoint> --graphs <graph-directory> --lut --install`: bind firmware/model ABI and update the sibling GUI manifest.
- `promote`, `final_exports`, `final_evaluations`: resume the existing frozen winner's remaining work only when explicitly requested. In the original workspace they reuse the existing `results/winner.json` and completed checkpoints. A fresh checkout must reconstruct the result workspace from release receipts and regenerate missing artifacts; source paths currently assume sibling Windows repositories under `Documents`.
- `summarize_final`: requires all 30 completed deployed-arithmetic evaluations before reporting acceptance.

The graph transformation/build scripts and detailed commands are in the sibling `Custom-H747XIH6` repository, `tools/` and `Docs/minigru-selected-reproduction-2026-10-04.md`. The matching selected model image and evaluation masks are in `BCI-STM32-Plot/data/ai_device_sessions/persistent/`. X-CUBE-AI 10.2.0 and STM32CubeIDE 1.19.0 were used. The existing graph remains available locally for rollback; generated operator profiling of the selected fallback and final memory/switch checks are still pending.
