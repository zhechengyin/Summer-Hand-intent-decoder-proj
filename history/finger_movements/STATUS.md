# FingerMovements Archive Status

Updated: 2026-08-10

The Phase 1h snapshot and the initial Phase A2/Phase 2b experiments were based
on a retired UEA conversion with deterministic adjacent-channel overlap. The
old 68.89% development score, 62.10% one-time test score, 59.81% Phase A2
score, and 63.41% Phase 2b score are preserved in this archive only to explain
the project history. They are not valid evidence for the corrected dataset.

After direct conversion of the official MATLAB release, TRAIN-only reruns
produced:

| Completed model | Mean OOF BA | Seed SD | Worst seed |
|---|---:|---:|---:|
| Terminal features + Logistic | 78.58% | 1.04 pp | 77.22% |
| Phase A2 CSSD + hierarchical LDA | 85.03% | 1.27 pp | 83.25% |
| Phase 2b winner | 86.72% | 0.68 pp | 86.09% |

The Phase 2b winner uses empirical covariance, per-trial trace normalization,
one F2 component per class, and LDA fusion. Its active implementation and new
all-TRAIN checkpoint are outside the archive under
`models/finger_movements/cssd_lda/`.

No archived code is an active dependency. The official TEST was not rerun
after the data correction and cannot be treated as a pristine final gate.
