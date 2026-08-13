# Archived Phase 2b Offline CSSD + LDA

This package preserves the strongest corrected-data offline reference.

- Empirical CSSD covariance.
- Per-trial trace normalization.
- One F2 spatial component per class.
- BP, ERD/F2, and BP-trend branch LDAs with final LDA fusion.
- Corrected TRAIN-only OOF BA: 86.72% mean, 0.68 pp seed SD, 86.09% worst seed.
- Checkpoint SHA-256: `1e95b1ab5eaf7277cadd658578ef343f67923fc2b197aec8e1231735163bbfa2`.

Its temporal filtering is zero-phase and therefore uses future samples relative to intermediate time points. It is a scientific offline reference, not the firmware candidate.
