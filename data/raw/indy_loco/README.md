# Immutable Indy raw recordings

The 37 original-name Indy MAT sessions from Zenodo 3854034 live under `indy/`.
The MAT payloads are immutable: processing code may only read them.

Aliases such as `train1` are resolved by `src.intent_decoder.data.indy` without
renaming source files. The official filename/checksum inventory is versioned in
`../../../configs/datasets/indy_sessions.yaml`.
