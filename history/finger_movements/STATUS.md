# FingerMovements Status at Phase 1d Completion

Archived: 2026-08-03

- Dataset: FingerMovements, 28 EEG channels, 50 samples per case.
- Development data: 316 official training cases.
- Locked data: 100 official test cases, never loaded during Phase 1.
- Validation: seeds 42, 43, and 44 with stratified five-fold cross-validation.
- Data-audit verdict: passed with unavailable trial-level session IDs recorded
  as a structural limitation.
- Selected candidate: 196 handcrafted features + L2 Logistic Regression.
- Candidate regularization: `C=1`, not yet final.
- Mean OOF balanced accuracy: 64.37%.
- Seed standard deviation: 1.50 percentage points.
- Worst-seed balanced accuracy: 62.68%.
- Final all-training-data checkpoint: not yet trained at archive time.

Active ownership moved to `models/finger_movements/feature_logistic/`. Phase 1e
must freeze `C` before final training or official-test evaluation.
