# Experiments

- `active/` contains only experiments capable of changing a current decision.
- `deepblue/` contains the separate U-M finger-SBP benchmark.

Retired numbered experiments and compatibility tools are summarized in
`docs/history/ARCHIVE_RETIREMENT.md`. A few recent superseded Indy scripts are
isolated under root `history/` for provenance; active code must not import them.

New reusable functions must go under `src/intent_decoder/`, not into another
numbered experiment. New experiment filenames should describe the question rather
than continue the global `iterNN` sequence.
