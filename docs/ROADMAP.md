# Roadmap

Work in this order; do not resume broad architecture sweeps before these items.

1. **Reproducible data inventory**
   - Complete `configs/datasets/indy_sessions.yaml` with checksums and local presence.
   - Add/download the missing month-CV sessions under `data/raw/indy_loco/`.
   - Place and verify Deep Blue separately under `data/raw/umich_deepblue/`.

2. **Causal preprocessing adoption**
   - Generate counts directly from raw events.
   - Add causal EWMA without centered Gaussian smoothing.
   - Fit normalization on training data or a fixed observation prefix, never the
     future portion of a held-out session.

3. **Correct drift-detector evaluation**
   - Use a fixed 60-second observation prefix.
   - Perform nested leave-one-month-out validation.
   - Select thresholds only inside training/validation folds.
   - Report sensitivity, specificity, false positives/negatives and uncertainty
     across seeds.

4. **Promote the 32-channel candidate**
   - Train the frozen selected configuration.
   - Save checkpoint, exact channel rule, normalization state and dataset manifest.
   - Export int8 and measure R² after quantization.
   - Measure actual STM32 RAM, flash, execution time and detector cost.

5. **Archive cleanup**
   - Once the new pipeline reproduces the required baselines, remove compatibility
     imports that archived `iter*` scripts still need.
   - Keep small metrics JSON and manifests versioned; keep large logs ignored.
