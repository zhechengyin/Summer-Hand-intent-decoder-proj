# Legacy — concluded work

Earlier lines of work, kept for provenance. **Not the active model** (that's
`../models/`). The EEG/fNIRS scripts are self-contained and compute their root as
this `legacy/` folder, so `import tools.X` and `from src import Y` resolve within
legacy without touching the active model package. Data paths point at
`legacy/data/…`.

## `eeg_fnirs_pipeline` — the original EEG+fNIRS intent decoder

- **`src/`** — staged pipeline (preprocess EEG/fNIRS, features, FBCSP, Riemannian
  tangent space, temporal CNN, SNN, fusion, N1/N2 replay).
- **`main.py`** — CLI orchestrator (`python main.py smoke|inspect|preprocess|
  train|evaluate|riemannian|…`).
- **`config.yaml`** — pipeline config.

Motor-imagery / action-category classification on ds004022, eegmmidb, and a
fused EEG+fNIRS set. Outcome: **near-chance** on same-limb MI; best LORO ~0.30.
Clustering recovered subject identity, not action class. Concluded.

## `tools/` — EEG experiment scripts

Probes and benchmarks behind the EEG results:

- **Motor imagery:** `eegmmidb_probe`, `mi_activation`, `architecture_bench`,
  `cursor_probe`, `muse_probe`, `two_stage`, `next_experiments`.
- **Riemannian / subject-structure:** `riemannian_cluster_probe`,
  `riemannian_subject_id_probe`, `aligned_riemannian_plot`, `eeg_only_ablation`.
- **EEG → velocity (WAY-EEG-GAL):** `way_gal_probe`, `way_gal_kinematics`,
  `way_gal_kin_models`, `way_gal_kin_full`, **`way_gal_kin_research`**.
- `download_data`, `tcn_gru_bench`, `convert_fnirs_octave.m`.

> **`way_gal_kin_research.py`** developed the TCN+GRU architecture now used by the
> current model. The shared pieces (`build_net`, `run_nn`, `run_linear`, `corr`,
> `BASE`) were extracted to `models/tcn_gru/best_model.py`; this file keeps its own full copy
> and its EEG-velocity experiment code.

## Best EEG result of record

EEG → hand/finger velocity (WAY-EEG-GAL), 3-subject mean **r = 0.853** (TCN+GRU).
This validated the architecture; the intracortical model is where it
reaches r ≈ 0.87 and became the active direction.

## `monkey_trials/` — concluded intracortical experiments

Architecture, activation, bin-width, smoothing, channel-count, and channel-
selection sweeps. These support the decisions captured in `../models/` but are
not active entry points.
