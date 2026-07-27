# Indy Results

This directory contains the retained evidence for the Indy 32-channel decoder.
Each folder name matches one experiment phase. Metrics and figures from the same
experiment stay together.

| Folder | Experiment | Contents |
| --- | --- | --- |
| `phase0a_data_audit/` | Dataset integrity and month-drift audit | Full audit JSON, per-session CSV, month summary, month-pair distances, and two figures |
| `phase0b_sampler_selection/` | Window-, session-, and month-balanced sampler comparison | Three-seed aggregate metrics and comparison figure |
| `phase1e_seed_crosscheck/` | Final hyperparameter cross-seed confirmation | Retained five-seed metrics supporting the frozen configuration |
| `phase2_locked_january/` | One-shot locked January evaluation | Final inference-only metrics and evaluation figure |
| `phase3a_drift_detector/` | Pre-January label-free detector baseline | Held-month scores, synthetic stress-test summary, figure and fitted reference artifact |
| `phase3b_leave_one_month_out/` | Strict out-of-month decoder evaluation | Five fold checkpoints, fold/session metrics and figure |
| `phase3c_decoder_state_detector/` | Decoder-derived second-layer gate | Session and sensitivity metrics, figure and active two-layer reference artifacts |
| `phase4a_architecture_sweep/` | Architecture-only pre-January Optuna study | Created when Phase 4a runs: study DB, ranked CSV, metrics, figure and disposable fold cache; no checkpoint |

## Phase mapping

- Phase 0a runner:
  `experiments/active/phase0a_data_audit.py`
- Phase 0b runner: deleted after its aggregate result was consolidated.
- Phase 1a--1e runners:
  `history/phase1/phase1a_optuna.py` through
  `history/phase1/phase1e_seed_crosscheck.py`
- Phase 2 runner:
  `history/phase2/phase2_locked_january.py`
- Phase 3a runner:
  `history/phase3/phase3a_drift_detector.py`
- Phase 3b runner:
  `history/phase3/phase3b_leave_one_month_out.py`
- Phase 3c runner:
  `history/phase3/phase3c_decoder_state_detector.py`
- Phase 4a runner:
  `experiments/active/phase4a_architecture_sweep.py`

Phase 1a--1d intermediate databases, figures, metrics, and non-selected
checkpoints were intentionally deleted. The retained Phase 1e metrics and
`models/indy_32ch/checkpoint.pt` are the authoritative model-selection
artifacts.

The January split is consumed. Do not use Phase 2 results to select another
model, feature set, detector threshold, or hyperparameter configuration.
Phase 3 is also development-selected and archived; the active execution path is
`models/indy_32ch/runtime.py`.

Phase 4a must not write a checkpoint. Its result is a shortlist for later
multi-seed confirmation, not a replacement for the frozen baseline.
