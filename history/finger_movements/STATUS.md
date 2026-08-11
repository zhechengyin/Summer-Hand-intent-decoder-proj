# FingerMovements Archive Status

Updated: 2026-08-11

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
| Phase 2c causal 500 ms / 50 ms candidate | 82.93% | 1.03 pp | 81.67% |
| Phase 2c causal 400 ms / 50 ms winner | 83.99% | 0.54 pp | 83.25% |

Phase 2d subsequently applied the frozen Phase 2c 400 ms checkpoint to the
corrected 100-case official TEST and achieved 77.05% balanced accuracy. This
is a retrospective benchmark, not a pristine blind test, because TEST was
exposed earlier in the project. It must not be used to retune the checkpoint.

The Phase 2b winner uses empirical covariance, per-trial trace normalization,
one F2 component per class, and LDA fusion. Its zero-phase implementation and
checkpoint are archived under `models/cssd_lda_offline_phase2b/`. The strictly
causal Phase 2c 500 ms baseline is archived under
`models/cssd_lda_causal_500ms_phase2c/`. The selected 400 ms successor and
all-TRAIN checkpoint are active outside this archive under
`models/finger_movements/cssd_lda/`.

No archived code is an active dependency. Phase 2d is preserved under
`experiments/phase2d_evaluate_frozen_test.py` and
`results/phase2d_official_test_400ms/`. Official TEST cannot be treated as a
pristine final gate.
