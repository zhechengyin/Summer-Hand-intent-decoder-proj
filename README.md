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
   needed — so the real fNIRS and fused branches work out of the box.
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
    ├── fbcsp.py           # Filter Bank CSP front-end for the EEG N1 (leak-safe)
    ├── window_sweep.py    # sweep analysis windows per modality, pick the best
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

# accuracy tuning:
python main.py sweep --modality eeg     # find the best EEG analysis window
python main.py sweep --modality fnirs   # fNIRS windows (accounts for HRF lag)
python main.py fbcsp                     # Filter Bank CSP N1 (subject + LOSO)

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
by UID. On this dataset real 4-class accuracy lands **near chance** (0.24–0.26;
see §7b) — same-limb MI is genuinely hard. The pipeline is validated on synthetic
data where it reaches 1.0, so the low real numbers reflect data difficulty.

## 7b. Accuracy techniques (window sweep + FBCSP)

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

**FBCSP** (`python main.py fbcsp`). Filter Bank Common Spatial Pattern: band-pass
into 8–12/12–16/16–20/20–24/24–30 Hz, learn **one-vs-rest CSP** spatial filters
per band, take log-variance features, select the most informative by mutual
information, then classify. CSP is supervised and cross-trial, so it is fit
**inside every CV fold** (no leakage). Knobs live under `fbcsp:` in the config.

**Honest finding across all 7 subjects.** With the full dataset and proper
held-out testing, 4-class *same-limb* motor imagery is **at chance** with these
EEG features — and that is a property of the task, not a bug:

| Method (all 7 subjects) | subject-specific | leave-one-run-out | LOSO |
|---|---|---|---|
| Bandpower + LDA (0–5 s)         | 0.239 | 0.224 | 0.256 |
| Bandpower + LDA (best win 1–5 s) | 0.257 | — | — |
| FBCSP (n_components = 1)         | 0.242 | 0.262 | 0.263 |

Chance = 0.25; the best single subject (sub-01) reaches only 0.30. A **binary
class-pair probe** (chance = 0.50) shows even the easiest 2-class contrast is
barely separable:

```
reach/grasp 0.44   reach/lift 0.49   reach/twist 0.50
grasp/lift  0.51   grasp/twist 0.52  lift/twist  0.53
```

So there is little linearly-separable same-limb MI signal in these bandpower/CSP
features. Adding data and *reducing model capacity* (fewer CSP components) nudge
LOSO a hair above chance but do not change the picture. This matches the dataset's
own framing — four movements of the **same** limb are far harder than the
spatially separated left-vs-right-hand MI most BCIs use — and these are patients
with orthopedic impairment, whose imagery may be weak or variable.

**What could actually move the needle (honest options, not guarantees):**
- **fNIRS fusion** — the hemodynamic channel is complementary; convert it
  (`tools/convert_fnirs_octave.m`) and fuse. Biggest untapped lever *in this
  dataset*.
- **Imagery-vs-rest** decoding using the `Rest Onset` markers — a 2-class
  move/rest contrast is usually far more decodable and would confirm the pipeline
  extracts real signal when a real contrast exists.
- **Riemannian / tangent-space** covariance features (`pyriemann`) sometimes beat
  bandpower/CSP (keep expectations modest given the binary probe).
- Treat this specific 4-class same-limb decode as near the dataset's ceiling and
  report it honestly.

The pipeline itself is correct: on synthetic data with separable classes both
bandpower and FBCSP reach 1.0, so the low real numbers reflect **data difficulty,
not implementation bugs**.

## 7c. Multimodal fusion (EEG + fNIRS)

ds004022's unique asset is *simultaneous* EEG (fast electrical) + fNIRS (slow
hemodynamic). The loader recovers the fNIRS signal straight from the MATLAB-table
subsystem in pure Python, so all three N1 modes run on **real** data:

| Modality (all 7 subjects) | subject-specific | leave-one-run-out | LOSO |
|---|---|---|---|
| EEG-only            | 0.239 | 0.224 | 0.256 |
| fNIRS-only          | 0.251 | 0.227 | 0.251 |
| **EEG + fNIRS fused** | 0.240 | 0.239 | 0.254 |

Chance = 0.25. Fusion is **feature-level** (concatenate EEG + fNIRS features →
one `StandardScaler` + classifier Pipeline), with trials aligned across the two
devices by matching each run's label sequence (robust to the occasional spurious
fNIRS marker). Consistent with the single-modality results, **fusion does not beat
chance here** — there is little separable 4-class same-limb signal in *either*
modality, so combining them cannot manufacture it (fused leave-one-run-out 0.239
edges out either alone, but within noise). What this delivers is a *correct,
real-data multimodal N1* and an honest three-way comparison — exactly the
"don't assume fusion wins" check — not an inflated number.

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

1. **fNIRS preprocessing depth** — the raw signal already loads in pure Python;
   add proper motion correction (TDDR / spline) + short-separation regression and
   verify the intensity/OD auto-detection against the acquisition metadata.
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
