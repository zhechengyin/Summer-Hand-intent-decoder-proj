# Repository collaboration guide

The repository contains two independent projects: `finger_movements/` and
`indy_loco/`. Shared root tooling may check both projects, but project code must
never import from the other project.

## Setup

```bash
make setup-dev
make lint
make test
```

Changes to the FingerMovements checkpoint, Python model, exporter, or C code
must also pass `make firmware-test`.

## Project boundaries

Within each project:

- `data/raw/` contains immutable downloaded sources;
- `data/processed/` contains reproducible model-ready arrays;
- `data/processing/` contains supported conversion code;
- `models/` contains retained implementations and checkpoints;
- `experiments/active/` contains at most one approved experiment;
- `results/` contains only that active experiment's outputs;
- `history/` contains completed immutable experiment provenance;
- `docs/STATUS.md` is the current technical source of truth;
- `tests/` protects that project's active artifacts and contracts.

Do not import code from `history/`. If an archived idea is reused, implement it
self-contained under the corresponding project's active directories.

## Protected artifacts

Do not modify raw data, archived evidence, frozen checkpoints, or official
test-derived metrics without an explicit project decision. A checkpoint change
must update its model decision, tests, status, and experiment log together.

## Experiment workflow

1. Agree on the question, protocol, phase name, metrics, and promotion rule.
2. Fit preprocessing inside training folds only.
3. Do not use an exposed official test set for model selection.
4. Record inputs, splits, seeds, metrics, checkpoint identity, and limitations.
5. After review, move the runner and evidence into that project's `history/`.

Do not create a new phase identifier without approval.

## Before pushing

```bash
make lint
make test
```

For FingerMovements firmware/model changes, also run:

```bash
make firmware-test
```

## Data sharing

Raw source data is never bundled. Archive only reproducible processed arrays:

```bash
make data-archive PROJECT=finger_movements
make data-archive PROJECT=indy_loco
```

Restore the archive from the repository root with `make data-restore` and the
same `PROJECT` and `DATA_ARCHIVE` values. Use checksums and an approved private
transfer channel.
