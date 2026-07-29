# Indy Loco Archive

Archived: 2026-07-29

This directory is the complete read-only snapshot of the retired Indy
intracortical hand-velocity decoder program. It preserves the previous research
state without allowing old code or assumptions to leak into the restarted
project.

## Archive contents

```text
STATUS.md                  pre-reset project status
EXPERIMENT_LOG.md          complete retained experiment record
PROJECT_README.md          pre-reset repository overview
EXPERIMENT_CODE_INDEX.md   index of the completed Phase-1 through Phase-4 code
requirements.txt           dependency snapshot
configs/                   dataset, model, and detector configuration
data/
  raw/                     37 immutable Indy MAT sessions
  processed/               29/4/4 model-ready NPZ split
  processing/              conversion notebook and causal target code
models/indy_32ch/          decoder, two-layer detector, runtime, and checkpoints
experiments/
  active_at_archive/       Phase-0a audit and Phase-5a runner at archive time
  phase1/ ... phase4/      completed selection and validation experiments
results/indy/              complete Phase-0 through Phase-5 result tree
tests/                     causality and session-balanced sampling tests
```

## Retained checkpoints

- `models/indy_32ch/64x64checkpoint.pt`: 32-channel 64/64 decoder tied to the
  archived integrated detector runtime.
- `models/indy_32ch/48x48checkpoint.pt`: smaller 32-channel firmware candidate.
- `results/indy/phase5a_64channel_width_comparison/checkpoints/`: experimental
  64-channel checkpoints retained with their Phase-5a evidence.

## Archive policy

- Do not run these files as active entry points.
- Do not import archive modules from new models or experiments.
- Do not modify the raw MAT files.
- Resolve old paths relative to the pre-reset layout documented in
  `PROJECT_README.md`; archived scripts may contain those historical paths.
- If any Indy work is resumed, restore it as a separate scoped project rather
  than mixing it into the new active model tree.
