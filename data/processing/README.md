# Data processing code

This directory contains scripts that convert immutable files from `../raw/` into
model-ready files under `../processed/`. Processing code is versioned; generated
NPZ files and run manifests are ignored by Git.

Keep dataset-specific scripts in their own folder. Reusable loaders and feature
functions still belong under `src/intent_decoder/`.
