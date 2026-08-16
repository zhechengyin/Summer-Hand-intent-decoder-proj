# FingerMovements

This project classifies left- versus right-finger movement from 28-channel EEG
sampled at 100 Hz. It is structurally and programmatically independent from the
Indy Loco project.

## Retained system

The firmware candidate is the frozen, strictly causal Phase 2c CSSD +
hierarchical LDA model. It uses a 400 ms past-only feature window, updates every
50 ms, and has a 100 ms cold-start filter pre-roll. All learned preprocessing
was selected from corrected official MATLAB TRAIN data only.

| Evidence | Balanced accuracy |
|---|---:|
| TRAIN-only mean OOF, seeds 42/43/44 | 83.99% |
| Worst seed OOF | 83.25% |
| Retrospective official TEST, 100 cases | 77.05% |

The official TEST result is retrospective because TEST had been exposed earlier
in the project. It must not be used for further model selection.

## Layout

```text
data/raw/FingerMovements/          immutable official source files
data/processed/finger_movements/   model-ready official splits
data/processing/finger_movements/  supported conversion code
models/cssd_lda/                   active model and frozen checkpoint
models/cssd_lda/firmware/          validated float32 C99 streaming port
experiments/active/                current experiment, presently empty
results/                           current result, presently empty
history/                           completed Phase 1–2f work and evidence
docs/STATUS.md                     technical source of truth
tests/                             frozen-artifact guardrails
```

Start with [`docs/STATUS.md`](docs/STATUS.md) for the detailed model state and
[`history/EXPERIMENT_LOG.md`](history/EXPERIMENT_LOG.md) for the decision trail.
