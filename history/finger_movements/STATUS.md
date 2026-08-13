# FingerMovements Archive Status

**Updated:** 2026-08-13

## Active result represented by this archive

The project currently deploys the strictly causal Phase 2c CSSD + hierarchical LDA model outside this archive at `../../models/finger_movements/cssd_lda/`.

| Item | Value |
|---|---|
| Task | left/right finger-movement classification |
| Source | official BCI Competition II Data Set IV MATLAB release |
| Input | 28 EEG channels, 100 Hz, isolated 500 ms cases |
| Development data | 316 TRAIN cases: 159 left, 157 right |
| Causal feature ring | previous 400 ms, after 100 ms cold-start filter pre-roll |
| Update interval | 50 ms |
| TRAIN-only OOF BA | 83.99% mean; 0.54 pp seed SD; 83.25% worst seed |
| Frozen official TEST BA | 77.05% on 100 cases; retrospective, not pristine |
| Checkpoint SHA-256 | `87b84cc2c8baf9efdc1ccf37ad28f5f58ad13c4db2a8f8a273fe73fce9956101` |

The apparent 89.89% all-TRAIN balanced accuracy is only a fit diagnostic, not a generalization estimate.

## Corrected TRAIN-only evidence

All comparisons below used five stratified folds for seeds 42, 43, and 44. Spatial filters, scalers, and classifiers were fitted inside each training fold; official TEST was not loaded.

| Model | Mean OOF BA | Seed SD | Worst seed | Decision |
|---|---:|---:|---:|---|
| Terminal features + Logistic | 78.58% | 1.04 pp | 77.22% | control only |
| Phase A2 CSSD + hierarchical LDA | 85.03% | 1.27 pp | 83.25% | baseline |
| Phase 2b zero-phase CSSD + LDA | 86.72% | 0.68 pp | 86.09% | best offline reference |
| Phase 2c causal 500 ms | 82.93% | 1.03 pp | 81.67% | superseded |
| Phase 2c causal 400 ms | 83.99% | 0.54 pp | 83.25% | active deployable model |
| Phase 2e regularized CSSD | 84.10% | 0.60 pp | 83.25% | not promoted |
| Phase 2e ToeplitzLDA | 84.50% | 0.90 pp | 83.23% | not promoted |
| Phase 2f Riemannian | 85.13% | 1.14 pp | 84.17% | not promoted |

## Invalid historical evidence

The former UEA conversion contained deterministic adjacent-channel overlap. Results generated from it—including 68.89% development BA, 62.10% one-time TEST BA, 59.81% Phase A2 BA, and 63.41% Phase 2b BA—are invalid for scientific comparison. They remain in the archive only for provenance.

## Deployment state

The active checkpoint has a self-contained float32 C99 port. Host validation covered all 316 TRAIN cases with zero class mismatches; maximum score error was `1.335e-4` and maximum probability error was `3.228e-5`. Runtime state is approximately 10,312 bytes and frozen constants approximately 903 bytes.

Continuous EEG is still required to validate long-running filter state, overlapping predictions, and rest/no-intent behavior. The official dataset contains isolated labeled epochs and no rest class.

## Archive rule

Phase 2e and Phase 2f did not satisfy their predefined promotion criteria. Do not substitute their artifacts for the active Phase 2c model. Official TEST has already been exposed and must not be used for further model selection.
