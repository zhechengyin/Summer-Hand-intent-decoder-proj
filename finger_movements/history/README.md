# FingerMovements Experiment Archive

This directory preserves completed and superseded FingerMovements work. Nothing here is an active Python dependency.

## Read this validity boundary first

Experiments run through 2026-08-07 used a retired UEA conversion later found to contain deterministic adjacent-channel overlap. Those results are retained only to explain project history and must not be compared with results from the corrected official MATLAB data.

Corrected evidence begins with the direct MATLAB conversion on 2026-08-10. Directory names containing `official_matlab`, plus Phase 2c and later results, use the corrected source unless explicitly marked otherwise.

## Final outcome

- Best offline zero-phase model: CSSD + hierarchical LDA, 86.72% mean out-of-fold balanced accuracy.
- Selected causal deployable model: 400 ms past-only CSSD + hierarchical LDA, 83.99% mean out-of-fold balanced accuracy.
- Retrospective official TEST result for the frozen causal model: 77.05% balanced accuracy on 100 cases.
- Phase 2e linear regularization and Phase 2f Riemannian alternatives were not promoted.
- The active model and firmware implementation live outside this archive at `../models/cssd_lda/`.

## Contents

- `STATUS.md`: current relationship between the archive and active model.
- `EXPERIMENT_LOG.md`: concise phase history, results, and decisions.
- `experiments/`: frozen runners.
- `models/`: superseded implementations and checkpoints.
- `results/`: metrics, predictions, and figures.

Old UEA result directories remain for provenance. Their invalid-source status is repeated in the relevant README and experiment log.
