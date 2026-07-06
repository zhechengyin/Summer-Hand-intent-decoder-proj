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
   `scipy`/`h5py` cannot deserialise. The loader recovers the channel columns
   **directly from the `.mat` subsystem in pure Python**
   (`extract_mcos_double_columns` in `src/load_bids.py`) — no MATLAB/Octave
   needed — so the real fused EEG+fNIRS branch works out of the box.
   (`tools/convert_fnirs_octave.m` remains as a fallback.)

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
SVM, Random Forest, Gradient Boosting (XGBoost optional), plus a simple optional
PyTorch GELU MLP (`--classifier gelu_nn`). Each lives inside a
`StandardScaler → classifier` **Pipeline** so scaling is fit on training folds
only. Small fused neural experiments are available explicitly:
`python main.py temporal-cnn`, `python main.py snn`,
`python main.py riemannian-snn`, and `python main.py windowed-riemannian-snn`.

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
    ├── fbcsp.py           # EEG FBCSP front-end fused with fNIRS
    ├── riemannian.py      # EEG tangent-space covariance front-end fused with fNIRS
    ├── temporal_cnn.py    # raw EEG temporal CNN branch fused with fNIRS
    ├── snn.py             # bare LIF SNN branch fused with fNIRS
    ├── window_sweep.py    # sweep analysis windows per modality, pick the best
    ├── fusion.py          # Stage 4: trial-aligned feature-level fusion
    ├── train_n1.py        # Stage 5: N1 decoder (probability output)
    ├── evaluate.py        # Stage 7: subject-specific + leave-one-run-out metrics
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

# 4) evaluate the fused EEG+fNIRS N1
python main.py evaluate                    # add --no-loro to skip run-held-out CV
python main.py evaluate --classifier svm   # try a different N1 baseline
python main.py evaluate --classifier gelu_nn  # simple GELU neural network

# 5) train and save a single N1 model
python main.py train
python main.py train --classifier gelu_nn

# 6) fused time-domain N1 -> N2 -> avatar demo (add --animate for a gif)
python main.py demo --max-trials 8

# accuracy tuning:
python main.py sweep --modality eeg     # find the best EEG analysis window
python main.py sweep --modality fnirs   # fNIRS windows (accounts for HRF lag)
python main.py fbcsp                     # EEG FBCSP + fNIRS N1
python main.py riemannian                # EEG tangent space + fNIRS N1
python main.py temporal-cnn              # raw EEG temporal CNN + fNIRS N1
python main.py snn                       # bare SNN + fNIRS N1
python main.py riemannian-snn            # tangent-space features + SNN head
python main.py windowed-riemannian-snn   # windowed tangent-space sequence + SNN

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
`preprocess_fnirs.py` picks up automatically; the fused EEG+fNIRS models then
become available.

## 7. Evaluation & how to read the results

* **Subject-specific** — stratified *K*-fold **within** each participant, then
  aggregated (the realistic first milestone for 7 subjects).
* **Leave-one-run-out** — within each participant, train on two runs and test on
  the held-out run.
* **Reported**: accuracy, **balanced accuracy**, **macro-F1**, per-class F1,
  confusion matrices (`results/figures/`), and fused-model comparison rows
  (`results/metrics/comparison.csv`).
* **Chance level = 0.25** (4 balanced classes). Judge everything against it.
* **N1-only vs N1+N2**: `metrics.json → *_n1_vs_n2` reports how often N2 defers
  (safety) or state-remaps vs a naive class→command map; the replay prints an
  accepted/deferred tally on a realistic continuous stream.

**Leakage safeguards:** non-overlapping trial epochs; the scaler lives inside the
Pipeline (fit on train folds only); leave-one-run-out groups by run within each
subject; fusion aligns trials by UID. On this dataset real 4-class accuracy lands
**near chance** (0.24–0.26; see §7b) — same-limb MI is genuinely hard.
The pipeline is validated on synthetic data where it reaches 1.0, so the low real
numbers reflect data difficulty.

## 7b. Accuracy techniques (window sweep + FBCSP + Riemannian + neural checks)

Two standard MI-BCI accuracy levers are built in.

**Window sweep** (`python main.py sweep --modality eeg|fnirs|both`). The best
analysis window is not obvious and is *modality-specific*: EEG mu/beta
(de)synchronisation is strongest **during** imagery (0–5 s / 0.5–4.5 s), while
fNIRS is hemodynamic and **lags** neural activity (response starts ~2 s, peaks
~5–6 s), so fNIRS windows are shifted/longer (2–7 s, 3–8 s, 4–10 s). The sweep
epochs one wide window, crops each candidate, and scores it; results land in
`results/metrics/window_sweep_<modality>.csv` + a bar chart. You can also enable
common-average reference with `eeg.car: true` in `config.yaml` (the dataset has
no EOG channels, so there is nothing extra to remove).

**FBCSP + fNIRS** (`python main.py fbcsp`). The EEG branch band-passes into
8–12/12–16/16–20/20–24/24–30 Hz, learns **one-vs-rest CSP** spatial filters per
band, takes log-variance features, and selects the most informative by mutual
information. Those EEG features are concatenated with aligned fNIRS features
before scaling/classification. CSP is supervised and cross-trial, so it is fit
**inside every CV fold** (no leakage). Knobs live under `fbcsp:` in the config.

**Riemannian + fNIRS** (`python main.py riemannian`). The EEG branch is filtered
to the motor-imagery band, converted to regularised channel covariance matrices,
and projected into tangent space at the training-fold covariance mean. Those EEG
tangent-space features are concatenated with aligned fNIRS features before
scaling/classification. The covariance mean and tangent projection are fit inside
each CV fold, just like CSP, so test trials do not influence the geometry. Knobs
live under `riemannian:` in the config.

**Temporal CNN + fNIRS** (`python main.py temporal-cnn`). The EEG branch consumes
the raw preprocessed EEG epoch with a small EEGNet-style temporal/spatial
convolution stack. The fNIRS branch consumes aligned hemodynamic features through
a small dense layer, then both embeddings are concatenated before classification.
The CNN, EEG normalization, and fNIRS scaler are fit from scratch inside each CV
fold. Knobs live under `temporal_cnn:` in the config.

**Bare SNN + fNIRS** (`python main.py snn`). The EEG branch provides decimated
time steps to a single leaky-integrate-and-fire hidden layer with surrogate
gradients and a linear readout. The aligned fNIRS feature vector is repeated at
each SNN time step. No convolutional front end is used. Knobs live under `snn:`
in the config.

**Riemannian-SNN + fNIRS** (`python main.py riemannian-snn` and
`python main.py windowed-riemannian-snn`). The static variant feeds one
tangent-space EEG vector plus fNIRS features to a repeated-current LIF SNN. The
windowed variant feeds a sequence of tangent-space vectors from 1 s overlapping
EEG windows, with fNIRS repeated across the sequence. Riemannian means are fit
inside each CV fold.

**Honest finding across all 7 subjects.** With the full dataset and proper
held-out testing, 4-class *same-limb* motor imagery is **near chance** even after
fusing EEG+fNIRS — and that is a property of the task, not a bug:

| Fused EEG+fNIRS method (all 7 subjects) | subject-specific | leave-one-run-out |
|---|---|---|
| Bandpower/hemodynamic + LDA | 0.240 | 0.239 |
| FBCSP EEG front end + fNIRS + LDA | 0.220 | 0.236 |
| Riemannian EEG front end + fNIRS + Logistic Regression | 0.248 | 0.272 |
| Raw EEG temporal CNN + fNIRS dense branch | 0.255 | 0.260 |
| Bare SNN + fNIRS | 0.254 | 0.263 |
| Static Riemannian-SNN + fNIRS | 0.251 | 0.244 |
| Windowed Riemannian-SNN + fNIRS | 0.243 | 0.246 |

Chance = 0.25; the best single subject (sub-01) reaches only 0.30. A **binary
class-pair probe** (chance = 0.50) shows even the easiest 2-class contrast is
barely separable:

```
reach/grasp 0.44   reach/lift 0.49   reach/twist 0.50
grasp/lift  0.51   grasp/twist 0.52  lift/twist  0.53
```

So there is little linearly-separable same-limb MI signal in these fused features.
This matches the dataset's own framing — four movements of the **same** limb are
far harder than the
spatially separated left-vs-right-hand MI most BCIs use — and these are patients
with orthopedic impairment, whose imagery may be weak or variable.

**What could actually move the needle (honest options, not guarantees):**
- **Better fNIRS preprocessing** — the hemodynamic channel is complementary, but
  simple baseline features may be too weak/noisy for this 4-class same-limb task.
- **Imagery-vs-rest** decoding using the `Rest Onset` markers — a 2-class
  move/rest contrast is usually far more decodable and would confirm the pipeline
  extracts real signal when a real contrast exists.
- **Riemannian / tangent-space tuning** — try classifier choices, covariance
  shrinkage, and subject-specific frequency windows (keep expectations modest
  given the binary probe).
- Treat this specific 4-class same-limb decode as near the dataset's ceiling and
  report it honestly.

The pipeline itself is correct: on synthetic data with separable classes both
bandpower and FBCSP reach 1.0, so the low real numbers reflect **data difficulty,
not implementation bugs**.

## 7c. Multimodal policy (EEG + fNIRS)

All N1 models are now **EEG+fNIRS models**. The default path concatenates EEG
bandpower features with fNIRS hemodynamic features. The FBCSP and Riemannian
commands use EEG-specific front ends first, then concatenate those EEG-derived
features with aligned fNIRS features before scaling and classification.

Fusion is feature-level, with trials aligned across the two devices by matching
each run's label sequence (robust to occasional spurious fNIRS markers). The
scaler and any supervised EEG front end are fit inside each training fold.

## 8. Limitations

* High-level **motor-imagery classification**, not muscle/joint/torque decoding.
* **Small dataset** (7 subjects) → limited generalisation; subject-specific and
  run-held-out results should be interpreted cautiously.
* **fNIRS is slow** (hemodynamics), so it uses a longer epoch window than EEG and
  contributes differently to fusion.
* The fNIRS Beer-Lambert conversion here is a **simplified baseline** (natural-log
  OD, compact extinction table, no advanced motion correction).
* **N2 is a state-machine with safety heuristics**, not a biological spinal model.
* The **avatar is a proof-of-concept** command interface, not a validated
  prosthetic controller.

## 9. Future work (after the baseline)

1. **fNIRS preprocessing depth** — the raw signal already loads in pure Python;
   add proper motion correction (TDDR / spline) + short-separation regression and
   verify the intensity/OD auto-detection against the acquisition metadata.
2. **Better EEG features/models**: tune FBCSP / Riemannian tangent-space and use
   temporal CNN / SNN branches only as carefully reported capacity checks.
3. **Late + learned fusion**: per-modality probabilities into a meta-classifier;
   attention over EEG/fNIRS.
4. **Sliding-window online decoding** with calibrated probabilities (Platt /
   temperature scaling) so N2's confidence gate is well-founded.
5. **Richer N2**: hysteresis, dwell-time, per-command cost/safety model, undo, and
   an explicit "rest" class.
6. **Transfer learning** across subjects (Euclidean/Riemannian alignment) if
   cross-subject generalisation becomes a project goal again.
7. **Real-time I/O** (LSL) to move from offline replay to a live loop; a proper 3-D
   arm/URDF avatar.

---
Built with mne, scikit-learn, numpy/scipy, matplotlib. Dataset: OpenNeuro
ds004022 (CC0). This repository is a research/education proof of concept.
