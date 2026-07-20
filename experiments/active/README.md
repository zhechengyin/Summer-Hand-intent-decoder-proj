# Active experiments

Current active question: improve the 32-channel causal decoder with
validation-only hyperparameter optimization after freezing session-balanced
training sampling, without touching the locked January test split.

`indy_month_drift_analysis.py` and `indy_month_drift_analysis.ipynb` are the
completed 37-session data audit. They verify raw/processed identity and schema,
profile every session, quantify month separation with permutation tests, and
write the reusable metrics under `results/metrics/`. The analysis does not
modify raw or processed data.

`drift_detector_month_cv.py` remains the drift-evaluation starting point. It
uses reusable code from `src/intent_decoder/`, a fixed 60-second observation
prefix for every split, a causal velocity target, and selects a detector
threshold only from inner validation sessions. No prediction during warm-up is
treated as valid. All 37 sessions are now present locally. The completed CPU
seed 42/43/44 sampling comparison selected session-balanced training in all
three seeds. The next model experiment is implemented in the self-contained
`models/indy_32ch/sweep_phase1_optuna.py`: an Optuna sweep of learning rate,
AdamW weight decay, and dropout with that sampler fixed. Detector threshold work
remains a separate nested-validation experiment. Run it from the repository root:

```bash
python models/indy_32ch/sweep_phase1_optuna.py
```
