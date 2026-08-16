# Archived Terminal-Feature Logistic Model

This was the final Phase 1 pipeline before the project switched to CSSD. It combined low-pass terminal samples and recent-window summaries with L2 logistic regression.

- Original UEA-derived development and TEST results are invalid.
- Corrected official-MATLAB TRAIN reevaluation: 78.58% mean OOF BA, 1.04 pp seed SD, 77.22% worst seed.
- Stored Phase 1 checkpoint SHA-256: `f8fca725c3b638219bbd734257cd958779e595add2fe1118e1e78689bc120047`; this checkpoint belongs to the retired source representation.

The corrected CSSD models were substantially stronger, so this model remains a control only.
