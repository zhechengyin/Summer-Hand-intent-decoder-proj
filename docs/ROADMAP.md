# Roadmap

Work in this order; do not resume broad architecture sweeps before these items.

1. **Reproducible data inventory**
   - Complete `configs/datasets/indy_sessions.yaml` with checksums and local presence.
   - Add/download the missing month-CV sessions under `data/raw/indy_loco/`.
   - Place and verify Deep Blue separately under `data/raw/umich_deepblue/`.

2. **Generate corrected causal data**
   - Run `data/processing/indy_loco/build_bin_40ms_causal_counts.py` after the
     required raw sessions are present.
   - Treat every earlier score as historical because centered filtering, central
     differences, or whole-session normalization were used in older paths.
   - Run `python -m unittest tests/test_causality.py -v` before every benchmark.

3. **Correct drift-detector evaluation**
   - Use a fixed 60-second observation prefix.
   - Perform nested leave-one-month-out validation.
   - Select thresholds only inside training/validation folds.
   - Report sensitivity, specificity, false positives/negatives and uncertainty
     across seeds.
   - Establish this leakage-free baseline before starting Optuna model sweeps.

4. **Promote the 32-channel candidate**
   - Train the frozen selected configuration.
   - Save checkpoint, exact channel rule, normalization state and dataset manifest.
   - Export int8 and measure R² after quantization.
   - Measure actual STM32 RAM, flash, execution time and detector cost.

5. **Archive cleanup**
   - Once the new pipeline reproduces the required baselines, remove compatibility
     imports that archived `iter*` scripts still need.
   - Keep small metrics JSON and manifests versioned; keep large logs ignored.
