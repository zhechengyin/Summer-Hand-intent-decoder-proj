# Repository collaboration guide

This repository contains an active firmware candidate and a large experiment
archive. New contributors should preserve the distinction between supported
code, immutable evidence, and local/generated data.

## First-time setup

Create and activate a virtual environment, then run:

```bash
make setup-dev
make lint
make test
```

`setup-dev` installs `requirements.txt` and `requirements-dev.txt`, then
installs both pre-commit and pre-push hooks. It does not download dataset
payloads. The Python version must be 3.10 or newer.

Useful commands:

```bash
make help
make format
make firmware-test
make clean
```

## Source-of-truth directories

```text
data/raw/                              immutable downloaded source data
data/processed/                        generated model-ready arrays
data/processing/                       supported conversion code
models/finger_movements/cssd_lda/      active model and frozen checkpoint
models/finger_movements/cssd_lda/firmware/ active C99 inference port
experiments/active/                    at most one approved active experiment
results/                               outputs of the active experiment only
history/finger_movements/              completed immutable experiment evidence
docs/STATUS.md                         current technical source of truth
```

Do not import code from `history/` into active code. If an old idea is reused,
write a self-contained active implementation so the dependency boundary stays
clear.

## Protected artifacts

Do not modify any of the following without an explicit project decision:

- files under `data/raw/`;
- archived experiment code or metrics under `history/`;
- the frozen Phase 2c checkpoint;
- official TEST-derived metrics;
- experiment phase names or the active phase identifier.

The active checkpoint is guarded by an automated SHA-256 test. A deliberate
replacement requires updating the model decision, checkpoint, tests, status,
and experiment log together. Do not update the expected hash merely to silence
a failed test.

## Experiment workflow

1. Discuss and approve the question, protocol, phase name, metrics, and
   promotion rule before implementation. Do not create a new phase letter or
   number unilaterally.
2. Put only the current runner in `experiments/active/` and only its generated
   evidence in `results/`.
3. Fit every learned preprocessing step inside the relevant training fold.
4. Do not use official TEST for feature, threshold, hyperparameter, or model
   selection. It has already been exposed and is retrospective evidence only.
5. Record seeds, folds, input timing, data hashes, metrics, numerical checks,
   resource estimates when relevant, and the resulting decision.
6. After review, move completed code/results to `history/finger_movements/`,
   update its indices, and leave `experiments/active/` clean.

## Code and review policy

Before pushing:

```bash
make lint
make test
```

For changes to the C exporter, parameters, streaming implementation, Python
model, or checkpoint, also run:

```bash
make firmware-test
```

Pre-commit checks active source/configuration for malformed YAML/TOML, merge
markers, large files, whitespace, debug statements, Ruff lint, and Ruff
formatting. Archived provenance and raw data are deliberately excluded from
mechanical rewriting. The pre-push hook runs the repository tests.

Keep commits focused. A pull request should state:

- what changed and why;
- which data and checkpoint were used;
- commands run and their results;
- whether metrics or firmware behavior changed;
- any remaining deployment or validity limitation.

## Data sharing

Raw source data is not committed or bundled by repository tooling. To share
the reproducible processed arrays with another authorized contributor:

```bash
make data-archive
```

This creates the ignored `data.tar.gz` containing only `data/processed/`.
Restore it at the repository root with:

```bash
make data-restore
```

Use a checksum and an approved private transfer channel for the archive. The
recipient should still retain the processed-data README and source-data
provenance. Never treat a restored archive as independent test evidence.
