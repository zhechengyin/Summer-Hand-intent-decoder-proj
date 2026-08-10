# Project Status

Updated: 2026-08-10

## Current state

FingerMovements Phase 2b is complete. The best corrected-data configuration is
now the promoted active **offline research model**:

```text
empirical CSSD covariance
per-trial trace normalization enabled
one BP spatial pattern per class
one ERD/F2 spatial pattern per class
BP + ERD + BP-trend branch LDAs
final LDA fusion
```

The implementation and verified all-TRAIN checkpoint are under
`models/finger_movements/cssd_lda/`. All completed experiment runners and
result directories have been moved to `history/finger_movements/`. There is no
active experiment at this boundary.

## Valid data contract

- source: official BCI Competition II Data Set IV MATLAB release;
- task: binary left/right movement classification;
- input: 28 EEG channels, 50 samples at 100 Hz per case;
- development split: 316 official TRAIN cases, left 159/right 157;
- processed TRAIN SHA-256:
  `a2025f277b5351839554e0ecf3398f1f4fd5151a4fc90f0e25c873734f5a91d1`;
- official TEST: previously opened during the invalid UEA-conversion phase and
  unavailable as a pristine selection gate.

The former UEA conversion had deterministic adjacent-channel overlap and is
retired. Its Phase 1, original Phase A2, isolated Phase 2b, and official-test
scores remain in history only as provenance; they are not valid evidence for
the corrected dataset.

## Corrected TRAIN-only evidence

All figures below use seeds 42/43/44 with five stratified folds per seed. Every
learned spatial filter, scaler, and classifier was fitted within its training
fold. Official TEST was not loaded.

| Model | Mean OOF BA | Seed SD | Worst seed |
|---|---:|---:|---:|
| Archived terminal features + Logistic, re-evaluated | 78.58% | 1.04 pp | 77.22% |
| Paper-style Phase A2 CSSD + hierarchical LDA | 85.03% | 1.27 pp | 83.25% |
| **Promoted Phase 2b configuration** | **86.72%** | **0.68 pp** | **86.09%** |

The Phase 2b winner improved all three seeds over the corrected Phase A2
reference. It was empirical covariance, trial trace normalization on, one F2
component per class, and LDA fusion. This supersedes the old invalid-data
conclusion that trial normalization should be off.

## Active model and checkpoint

Implementation:

```text
models/finger_movements/cssd_lda/model.py
```

Checkpoint:

```text
models/finger_movements/cssd_lda/checkpoints/finger_movements_cssd_lda_phase2b.npz
```

Checkpoint SHA-256:

```text
1e95b1ab5eaf7277cadd658578ef343f67923fc2b197aec8e1231735163bbfa2
```

The checkpoint was fitted once on all 316 official TRAIN cases, saved,
reloaded, and verified with zero score/probability error and identical
predictions. Apparent fitting balanced accuracy was 90.84%; this is a fit
diagnostic, not a held-out estimate. Official TEST was refused and not loaded.

## Deployment boundary

The selected model uses fourth-order zero-phase Butterworth filtering. That
operation uses future samples within each 500 ms trial, so this checkpoint is
not a causal streaming firmware model. It must not be described as real-time
deployable yet.

The next justified phase is to create a causal preprocessing replacement,
repeat TRAIN-only cross-validation against the frozen 86.72% offline
reference, and measure the accuracy cost before exporting coefficients to
firmware. A new external holdout is required for a genuinely independent
final performance claim.
