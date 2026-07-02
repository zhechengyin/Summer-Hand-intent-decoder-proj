# Neural Intent Decoder — a two-stage (N1 + N2) upper-limb motor-imagery pipeline

Decode **imagined right-upper-limb actions** — **REACH / GRASP / LIFT / TWIST** —
from EEG + fNIRS biosignals (OpenNeuro **ds004022**) and turn them into
**safe, state-aware prosthetic/avatar commands**.

> **Framing / honesty.** This is a *high-level motor-imagery intent decoder*, not
> a biomechanical prosthetic controller. It classifies the **category** of an
> intended movement and issues a **high-level command** (e.g. `close_hand`). It
> does **not** decode muscle activations, joint angles, torque, or finger
> kinematics. The avatar is a proof-of-concept command interface, and N2 is a
> command-interpretation state machine with safety heuristics — **not** a
> validated model of the spinal cord.

---

## 1. Project goal

| Milestone | Deliverable |
|-----------|-------------|
| **1** | Reliable classification of the imagined movement category (4-class). |
| **2** | Use those predictions to drive a simple prosthetic-arm / avatar simulation over time. |

The system is deliberately built as an **online-style loop** (windows arriving
over time → decode → interpret → act), even though v1 replays offline trials.

## 2. Dataset — OpenNeuro ds004022

*"Multimodal EEG and fNIRS Biosignal Acquisition during Motor Imagery Tasks in
Patients with Orthopedic Impairment"* (Lee et al., CC0).

| | EEG | fNIRS |
|---|---|---|
| Format | EEGLAB `.set`/`.fdt` | **BBCI toolbox `.mat`** (`nirs_data`) |
| Channels | 18 (10-10, ref FCz) | 8 sources × 4 detectors, λ = 760/850 nm |
| Rate | 500 Hz | 7.8125 Hz |
| Line noise | 60 Hz | — |
| Subjects / runs | 7 / 3 | 7 / 3 |
| Trials per run | 40 (10 per class) | 40 (10 per class) |

**Trial timeline (15 s):** 3 s fixation → 4 s visual cue → 3 s ready →
**5 s motor imagery**.

**Two realities that differ from a "textbook" BIDS dataset** (handled in code):

1. **No `events.tsv`.** Labels are embedded in the EEGLAB event struct
   (BrainVision markers `S 3/4/5/6` = Reach/Grasp/Lift/Twist, `S 8` = imagery
   onset) and in the fNIRS BBCI marker struct (`mrk.toe` codes 3–6). The loader
   reads them from annotations, not TSVs.
2. **fNIRS raw signal is a MATLAB `table` (MCOS object)** inside `cnt.x`, which
   `scipy`/`h5py` cannot deserialise. Markers + montage load fine; the intensity
   matrix needs a one-time conversion (`tools/convert_fnirs_octave.m`). Until
   then the EEG pipeline and the synthetic demo run fully, and the fNIRS branch is
   skipped with a clear message.

See [`data/README.md`](data/README.md) for the download options and the caveat.

## 3. AI-Spine architecture

```
 EEG/fNIRS ─► preprocess ─► features ─►  N1  ─► p_t ─►  N2  ─► command u_t ─► avatar
 (x_t)        (filter/epoch) (bandpower/   (evidence  (state-      (state-      (state
                              hemodynamic)  decoder)   injected)    aware)       s_{t+1})
                                              ▲                        ▲
                                        probability vector       current state s_t
```

### N1 — Neural Evidence Decoder (`src/train_n1.py`)
Time-domain windows → **probability vector** over the four actions, plus a
**confidence**, a **margin** (top1−top2) and a normalised **entropy**.

```json
{ "reach": 0.10, "grasp": 0.72, "lift": 0.08, "twist": 0.10 }
```
Baselines (small dataset ⇒ classical first): LDA (default), Logistic Regression,
SVM, Random Forest, Gradient Boosting (XGBoost optional). Each lives inside a
`StandardScaler → classifier` **Pipeline** so scaling is fit on training folds
only. An optional temporal-CNN is *proposed* (`build_torch_cnn`) but off by
default.

### N2 — State-Injected Intent-to-Command Interpreter (`src/mini_ai_spine_n2.py`)
Takes N1's probabilities **and the current avatar state** and returns a safe
command:

```json
{ "intent": "GRASP", "confidence": 0.72,
  "current_state": { "hand_state": "open", "holding_object": false },
  "prosthetic_action": "close_hand",
  "command_vector": { "reach": 0.10, "grasp": 0.72, "lift": 0.08, "twist": 0.10 } }
```

It applies **confidence + margin gating**, **majority-vote smoothing** over
recent windows, **hold/no-action** when uncertain, and **state-aware rules**:

| Intent | Base command | State-aware refinement |
|--------|--------------|------------------------|
| Reach  | `extend_arm_forward` | `hold_reach` if already fully extended |
| Grasp  | `close_hand` | `maintain_grip` / `increase_grip_stability` if already holding |
| Lift   | `raise_arm` | `hold_position` if already raised |
| Twist  | `rotate_wrist` | `limit_rotation` / `block_rotation` at the safe wrist limit |
| (low confidence) | — | `hold_state` / `request_more_evidence` |

### Time-domain inference loop (`src/replay_time_domain.py`)
```
for each time step t:
    x_t   = next signal window          s_t   = current avatar state
    p_t   = N1(x_t)                      u_t   = N2(p_t, s_t)
    apply u_t → s_{t+1}
```
Each trial is sliced into several sub-windows so N1 emits a *stream* of vectors
and N2's smoothing/gating genuinely matters.

## 4. Project structure

```
neural_intent_decoder/
├── README.md            config.yaml         requirements.txt      main.py   (CLI)
├── data/                # ds004022 goes here (git-ignored) + download notes
├── notebooks/
├── tools/
│   ├── download_data.py            # openneuro-py / aws s3 wrapper
│   └── convert_fnirs_octave.m      # MCOS-table -> plain-array fNIRS converter
├── results/
│   ├── figures/         # confusion matrices, replay.gif
│   └── metrics/         # metrics.json, comparison.csv, replay_log.csv
└── src/
    ├── config.py          # load config.yaml, seeding, paths
    ├── containers.py      # TrialEpochs (the array container between stages)
    ├── load_bids.py       # Stage 1: discover files, load EEG(.set)/fNIRS(.mat)
    ├── preprocess_eeg.py   # Stage 2: notch/band-pass/epoch EEG
    ├── preprocess_fnirs.py # Stage 2: OD -> Beer-Lambert HbO/HbR -> epoch
    ├── feature_extraction.py # Stage 3: bandpower (EEG) + hemodynamic (fNIRS)
    ├── fusion.py          # Stage 4: trial-aligned feature-level fusion
    ├── train_n1.py        # Stage 5: N1 decoder (probability output)
    ├── evaluate.py        # Stage 7: subject-specific + LOSO, metrics, figures
    ├── state.py           # Stage 6: ProstheticState
    ├── mini_ai_spine_n2.py # Stage 6: N2 state-injected interpreter
    ├── simulate_avatar.py  # Stage 9: render the arm (ASCII + matplotlib)
    ├── replay_time_domain.py # Stage 8: the online-style replay loop
    └── synthetic.py       # class-structured synthetic data for smoke tests
```

## 5. Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python ≥ 3.10. Core deps: numpy, scipy, pandas, matplotlib, scikit-learn, mne,
PyYAML (PyTorch/XGBoost optional).

## 6. Usage

```bash
# 0) (optional) try the WHOLE pipeline with no download — synthetic data
python main.py smoke

# 1) get the data (see data/README.md for aws/datalad options)
python tools/download_data.py --target data/ds004022

# 2) inspect the BIDS tree, event codes and trial counts
python main.py inspect

# 3) preprocess + cache epochs
python main.py preprocess

# 4) evaluate every available modality (EEG / fNIRS / fused), subject-specific + LOSO
python main.py evaluate                    # add --no-loso to skip cross-subject
python main.py evaluate --classifier svm   # try a different N1 baseline

# 5) train and save a single N1 model
python main.py train --modality eeg

# 6) time-domain N1 -> N2 -> avatar demo (add --animate for a gif)
python main.py demo --max-trials 8

# everything on the real data
python main.py all
```
Handy flags: `--subjects sub-01 sub-02`, `--no-cache`, `--config myconfig.yaml`.
All knobs (bands, epoch windows, thresholds, classifier) live in
[`config.yaml`](config.yaml).

### fNIRS: enabling the real signal
```bash
octave --no-gui tools/convert_fnirs_octave.m data/ds004022   # or MATLAB
```
This writes `*_nirs_converted.mat` (plain arrays) next to each fNIRS file, which
`preprocess_fnirs.py` picks up automatically; the fNIRS and fused modes then
become available.

## 7. Evaluation & how to read the results

* **Subject-specific** — stratified *K*-fold **within** each participant, then
  aggregated (the realistic first milestone for 7 subjects).
* **Leave-one-subject-out (LOSO)** — train on *n−1* subjects, test on the held-out
  one (the harder cross-subject stretch goal).
* **Reported**: accuracy, **balanced accuracy**, **macro-F1**, per-class F1,
  confusion matrices (`results/figures/`), and a EEG/fNIRS/fused comparison
  (`results/metrics/comparison.csv`).
* **Chance level = 0.25** (4 balanced classes). Judge everything against it.
* **N1-only vs N1+N2**: `metrics.json → *_n1_vs_n2` reports how often N2 defers
  (safety) or state-remaps vs a naive class→command map; the replay prints an
  accepted/deferred tally on a realistic continuous stream.

**Leakage safeguards:** non-overlapping trial epochs; the scaler lives inside the
Pipeline (fit on train folds only); LOSO groups by subject; fusion aligns trials
by UID. Expect **subject-specific ≫ LOSO** and real 4-class accuracy in the modest
range (well above 0.25 but far from perfect) — same-limb MI is genuinely hard.

## 8. Limitations

* High-level **motor-imagery classification**, not muscle/joint/torque decoding.
* **Small dataset** (7 subjects) → limited generalisation; LOSO will lag
  subject-specific.
* **fNIRS is slow** (hemodynamics), so it uses a longer epoch window than EEG and
  contributes differently to fusion.
* The fNIRS Beer-Lambert conversion here is a **simplified baseline** (natural-log
  OD, compact extinction table, no advanced motion correction).
* **N2 is a state-machine with safety heuristics**, not a biological spinal model.
* The **avatar is a proof-of-concept** command interface, not a validated
  prosthetic controller.

## 9. Future work (after the baseline)

1. **fNIRS conversion in-pipeline** (drop the Octave step) and proper motion
   correction (TDDR / spline) + short-separation regression.
2. **Better EEG features/models**: CSP / Riemannian tangent-space, FBCSP; then the
   temporal-CNN branch once augmentation/more data support it.
3. **Late + learned fusion**: per-modality probabilities into a meta-classifier;
   attention over EEG/fNIRS.
4. **Sliding-window online decoding** with calibrated probabilities (Platt /
   temperature scaling) so N2's confidence gate is well-founded.
5. **Richer N2**: hysteresis, dwell-time, per-command cost/safety model, undo, and
   an explicit "rest" class.
6. **Transfer learning** across subjects (Euclidean/Riemannian alignment) to lift
   LOSO performance.
7. **Real-time I/O** (LSL) to move from offline replay to a live loop; a proper 3-D
   arm/URDF avatar.

---
Built with mne, scikit-learn, numpy/scipy, matplotlib. Dataset: OpenNeuro
ds004022 (CC0). This repository is a research/education proof of concept.
