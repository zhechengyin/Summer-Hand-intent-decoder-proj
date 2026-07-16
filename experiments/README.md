# Experiments

- `active/` contains only experiments capable of changing a current decision.
- `archive/indy/` contains completed numbered experiments. They are retained for
  provenance and may use compatibility imports.
- `deepblue/` contains the separate U-M finger-SBP benchmark.
- `common/` contains compatibility helpers for archived scripts.
- `tools/` contains optional instrumentation.

New reusable functions must go under `src/intent_decoder/`, not into another
numbered experiment. New experiment filenames should describe the question rather
than continue the global `iterNN` sequence.
