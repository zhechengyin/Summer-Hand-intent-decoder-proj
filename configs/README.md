# Configs

There is no separate active configuration file. The frozen FingerMovements
Feature + Logistic input and candidate-training contracts live beside the model
under
[`models/finger_movements/feature_logistic/`](../models/finger_movements/feature_logistic/README.md),
so there is only one active source of truth. Logistic `C=1` is not final until
Phase 1e nested cross-validation is complete.

Retired Indy configurations are preserved under `history/indy/configs/`.
