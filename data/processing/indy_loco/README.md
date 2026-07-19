# Indy preprocessing

The supported Indy workflow is the reader-facing notebook under `indy/`. It
audits all 37 official sessions, explains the raw HDF5 structure, verifies the
Zenodo checksums, and writes model-ready causal artifacts to
`data/processed/indy_loco/indy/`.

The notebook is intentionally narrow and English-only. It stores 96-channel M1
counts as the sole model input, two-axis velocity as the label, and only minimal
provenance. It contains no exploratory plots or model-training code.

The raw directory is read-only. Channel selection, normalization, and model
features remain training-time operations so preprocessing cannot inspect a
session's scored future.
