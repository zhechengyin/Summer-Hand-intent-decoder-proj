# FingerMovements Status at Phase 1h Completion

Archived: 2026-08-05

- Dataset: FingerMovements, 28 EEG channels, 50 samples per case at 100 Hz.
- Development data: 316 official training cases.
- Validation: seeds 42, 43, and 44 with stratified five-fold cross-validation.
- Selected representation: 252 terminal causal low-pass ABC features.
- Selected classifier: L2 Logistic Regression with `C=1`.
- Mean OOF balanced accuracy: 68.89%.
- Seed standard deviation: 0.92 percentage points.
- Worst-seed balanced accuracy: 68.35%.
- Final checkpoint: trained on all 316 official training cases and verified
  after reload.
- Checkpoint SHA-256:
  `f8fca725c3b638219bbd734257cd958779e595add2fe1118e1e78689bc120047`.
- Official test: opened once for pure inference on 2026-08-05.
- Official-test accuracy: 62.00%.
- Official-test balanced accuracy: 62.10%.
- Official-test macro-F1: 61.94%.
- Official-test confusion matrix: `[[33, 16], [22, 29]]`.

The pipeline is a completed reproducible baseline, not an active final model.
Future work must not tune from the opened official-test result. The next
candidate direction should be selected entirely on the 316-case training
split and should focus on EEG temporal-frequency and spatial representation.
