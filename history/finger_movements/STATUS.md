# FingerMovements Status at Phase 1c Completion

Archived: 2026-08-03

- Dataset: FingerMovements, 28 EEG channels, 50 samples per case.
- Development data: 316 official training cases.
- Locked data: 100 official test cases, never loaded during Phase 1.
- Validation: seeds 42, 43, and 44 with stratified five-fold cross-validation.
- Selected pipeline: Feature + Linear.
- Selected duration: 50 epochs.
- Mean OOF balanced accuracy: 60.05%.
- Seed standard deviation: 0.36 percentage points.
- Worst-seed balanced accuracy: 59.84%.
- Final all-training-data checkpoint: not yet trained at archive time.

Active ownership moved to `models/finger_movements/feature_linear/`.
