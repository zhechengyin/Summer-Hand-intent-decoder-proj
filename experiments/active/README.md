# Active experiments

Current active question: validate the 32-channel drift detector without
whole-session normalization, same-month epoch-selection leakage, or a post-hoc
threshold.

`drift_detector_month_cv.py` is the new starting point. It uses reusable code from
`src/intent_decoder/`, a fixed 60-second observation prefix for every split, a
causal velocity target, and selects a detector threshold only from inner
validation sessions. No prediction during warm-up is treated as valid. It has not
yet been run because most raw month-CV sessions are not present locally.
