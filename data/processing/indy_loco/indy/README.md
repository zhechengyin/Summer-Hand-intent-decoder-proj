# Indy notebook

Open `prepare_indy_model_ready.ipynb` and run it from top to bottom with the
project virtual environment as the kernel.

The notebook reads the 37 untouched MAT files under
`data/raw/indy_loco/indy/` and writes per-session NPZ artifacts plus an inventory
and manifest under `data/processed/indy_loco/indy/`. Artifacts are placed in the
versioned chronological split: 29 train, 4 validation, and 4 locked test sessions.
It deliberately does not copy waveform snippets or fit channel
selection/normalization state.
