# FingerMovements Status at Phase 1f Completion

Archived: 2026-08-04

- Dataset: FingerMovements, 28 EEG channels, 50 samples per case.
- Development data: 316 official training cases.
- Locked data: 100 official test cases, never loaded during Phase 1.
- Validation: seeds 42, 43, and 44 with stratified five-fold cross-validation.
- Data-audit verdict: passed with unavailable trial-level session IDs recorded
  as a structural limitation.
- Selected candidate: 252 terminal low-pass features + L2 Logistic Regression.
- Frozen regularization: `C=1`.
- Mean OOF balanced accuracy: 68.89%.
- Seed standard deviation: 0.92 percentage points.
- Worst-seed balanced accuracy: 68.35%.
- Final all-training-data checkpoint: not yet trained at archive time.

Active ownership moved to `models/finger_movements/terminal_logistic/`. The
official test remained locked throughout Phase 1b–1f.
