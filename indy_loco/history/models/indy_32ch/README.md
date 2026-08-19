# Indy Causal Decoder Modules and Retained 32-Channel Models

This package contains the causal TCN+GRU implementation, retained 32-channel
checkpoints, and historical detector modules. Phase 6 did not replace these
artifacts; its promoted 96-channel model is isolated under `../indy_96ch/`.

## Checkpoints

| File | Architecture | Parameters | Intended use |
|---|---|---:|---|
| `48x48checkpoint.pt` | TCN width 48, GRU width 48 | 45,266 | preferred standalone firmware candidate |
| `64x64checkpoint.pt` | TCN width 64, GRU width 64 | 78,786 | reference model and detector reproduction |

Both models use 32 raw-count channels plus 32 causal EWMA features over 50 past bins. The 48/48 checkpoint is smaller and passed five-seed non-inferiority checks. The saved Phase 3 detector references use 64/64 internal states and cannot be attached to 48/48 without recalibration.

Key modules:

- `features.py` and `input_pipeline.py`: causal feature construction and data loading.
- `model.py`: TCN+GRU architecture.
- `sampling.py`: session-balanced sampling.
- `runtime.py`: decoder runtime path.
- `drift_detector.py` and `decoder_state_detector.py`: archived detector layers.

The detector modules and saved 32-channel weights remain historical. The
general decoder modules are retained for reproducibility; completed Phase 6
and Phase 7 runners are archived under `../../history/experiments/`.
FingerMovements remains an independent project.
