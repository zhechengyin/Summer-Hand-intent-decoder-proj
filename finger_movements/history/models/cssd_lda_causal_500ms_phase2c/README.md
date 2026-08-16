# Archived Phase 2c Causal 500 ms Model

This is the strictly causal 500 ms predecessor of the selected 400 ms model.

- Input: 28 channels at 100 Hz.
- History: 500 ms ending at the current prediction point; no future samples.
- Streaming update: 50 ms with persistent causal filter state.
- Corrected TRAIN-only OOF BA: 82.93% mean, 1.03 pp seed SD, 81.67% worst seed.
- Checkpoint SHA-256: `d92c23f7e6f8722d568d1b31963eab1328d5367ba32764b676d1ae0d73aaefd4`.

It was superseded because the 400 ms model reached 83.99% mean BA with better worst-seed performance and lower seed variability. This package is not an active dependency.
