# Roadmap

Work in this order; do not resume broad architecture sweeps before these items.

1. **Reproducible data inventory**
   - Indy complete: all 37 official sessions are present under
     `data/raw/indy_loco/indy/`, with Zenodo checksums recorded in
     `configs/datasets/indy_sessions.yaml`.
   - Place and verify Deep Blue separately under `data/raw/umich_deepblue/`.

2. **Generate corrected causal data — complete**
   - `data/processing/indy_loco/indy/prepare_indy_model_ready.ipynb` has produced
     37/37 valid outputs under
     `data/processed/indy_loco/indy/{train,validation,test}/` with counts 29/4/4.
   - Treat every earlier score as historical because centered filtering, central
     differences, or whole-session normalization were used in older paths.
   - Run `python -m unittest tests/test_causality.py -v` before every benchmark.

3. **Freeze training sampling — complete**
   - CPU seeds 42/43/44 compared window-, session-, and month-balanced sampling
     under identical initialization and training budgets.
   - Session-balanced won minimum pooled December validation loss in 3/3 seeds
     and is now the only active sampler.
   - January test remained locked and unloaded.

4. **Tune optimization and regularization on validation only — Phase 1 complete**
   - The CPU seed-42 study completed 40 trials with 29 complete, 11 pruned and
     zero failed. Trial 32 reached validation loss 0.482066 and R² 0.557579.
   - Run a small new-study boundary refinement because all top-five trials used
     the minimum searched dropout, 0.10. Do not change ranges inside the completed
     Phase-1 SQLite study.
   - Preserve session-balanced sampling and pooled December validation loss as
     the primary objective; retain macro and worst-session R² as guardrails.
   - Confirm a small set of distinct finalists across seeds 42/43/44 rather than
     spending three-seed budget on five nearly identical configurations.
   - Do not load January test data.

5. **Correct drift-detector evaluation**
   - Use a fixed 60-second observation prefix.
   - Perform nested leave-one-month-out validation.
   - Select thresholds only inside training/validation folds.
   - Report sensitivity, specificity, false positives/negatives and uncertainty
     across seeds.
   - Keep decoder hyperparameter selection and detector-threshold selection as
     separate experiments so neither reuses January test information.

6. **Promote the 32-channel candidate**
   - Train the frozen selected configuration.
   - Save checkpoint, exact channel rule, normalization state and dataset manifest.
   - Export int8 and measure R² after quantization.
   - Measure actual STM32 RAM, flash, execution time and detector cost.

7. **Version evidence and deploy**
   - Keep small metrics JSON and manifests versioned; keep large logs ignored.
   - Treat retired experiment directions as documentation only. Reimplement any
     revived idea against the supported causal API and nested protocol.
