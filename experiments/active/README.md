# Active experiments

Current active question: measure month-to-month Indy drift, then validate the
32-channel causal decoder under that drift without whole-session normalization,
same-month epoch-selection leakage, or a post-hoc threshold.

`indy_month_drift_analysis.py` and `indy_month_drift_analysis.ipynb` are the
completed 37-session data audit. They verify raw/processed identity and schema,
profile every session, quantify month separation with permutation tests, and
write the reusable metrics under `results/metrics/`. The analysis does not
modify raw or processed data.

`drift_detector_month_cv.py` is the new starting point. It uses reusable code from
`src/intent_decoder/`, a fixed 60-second observation prefix for every split, a
causal velocity target, and selects a detector threshold only from inner
validation sessions. No prediction during warm-up is treated as valid. All 37
sessions are now present locally; the next model experiment is month-aware
cross-validation with session- or month-balanced training batches.
