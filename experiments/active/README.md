# Active experiments

Current active question: validate the 32-channel drift detector without
whole-session normalization, same-month epoch-selection leakage, or a post-hoc
threshold.

`drift_detector_month_cv.py` is the new starting point. It uses reusable code from
`src/intent_decoder/`, a fixed 60-second observation prefix, and selects a detector
threshold only from inner validation sessions. It has not yet been run in this
checkout because most raw month-CV sessions are not present locally.
