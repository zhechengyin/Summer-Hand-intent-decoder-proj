# Phase 3 — Cross-Month Robustness and Drift Detection

These scripts preserve the retired detector investigation:

- `phase3a_drift_detector.py`: label-free 60-second signal statistics.
- `phase3b_leave_one_month_out.py`: strict pre-January decoder evaluation with one month held out at a time.
- `phase3c_decoder_state_detector.py`: frozen-decoder hidden/output compatibility layer.
- `test_*.py`: local protocol and implementation checks used during development.

The final retrospective result was that the second layer identified the June 30 and October 13 leave-month failures while passing the other 31 sessions. It was not prospectively validated, and its reference artifacts are specific to the archived 64/64 decoder.

Primary outputs are under `../../results/indy/phase3a_drift_detector/`, `phase3b_leave_one_month_out/`, and `phase3c_decoder_state_detector/`.
