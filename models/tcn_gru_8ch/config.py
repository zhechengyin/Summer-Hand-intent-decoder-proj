"""8-channel STM32 decoder — exact spec of record (importable).

The deployable decoder: 8 spike-detection channels -> 2D fingertip velocity via
the same dilated-TCN + bidirectional-GRU architecture as `models.tcn_gru`
(`build_net`), but shrunk to STM32 size and trained on more sessions. int8 form
is ~27 kB with no R² loss (LOG-054).

Hardware: 8 channels (threshold-crossing spike detection), STM32-class MCU.
Metric: R² (coefficient of determination, avg of X,Y velocity).
"""
from models.tcn_gru.best_model import BASE

# --- preprocessing (per-electrode pipeline; identical to the 96ch model) ---
BIN_S = 0.04                 # 40 ms bins -> 25 Hz
WINDOW_S = 2.0               # 2 s sequence windows
VEL_LOWPASS_HZ = 3.0         # low-pass finger position before differentiating (LOG-030)
RATE_SMOOTH_SIGMA_BINS = 1.0 # Gaussian smoothing of firing rates (LOG-032)
N_CHANNELS = 8               # hardware limit; top-8 electrodes by firing rate on train1-6
N_OUT = 2                    # 2D: top-2 velocity-variance movement axes

# Channel selection is FIXED to the top-8 firing electrodes of the original 6
# training sessions. Re-selecting on more data OVERFITS (0.628 -> 0.502, LOG-053);
# learned/corr selection also lose to firing-rate (LOG-043/046). Do not re-select.

# --- model hyperparameters (merged over BASE) ---
# 'wide' (F64/H64/L1): at 24 sessions more capacity finally helps (0.655->0.677,
# LOG-056). At 6 sessions the smaller F32/H32 was better (bigger overfit).
MODEL = {**BASE, "dils": [1, 2, 4, 8], "H": 64, "L": 1, "F": 64,
         "epochs": 60, "noise": 0.1, "chdrop": 0.1, "cosine": True,
         "act": "relu", "n_out": N_OUT}

# --- training data: 24 indy sessions (6 base split + 18 nearby) ---
# More data is the main R² lever and still climbing (6->18->24 sess =
# 0.529->0.628->0.655, LOG-052/053/055). A 3-seed ensemble adds ~+0.02 (0.675).
BASE_TRAIN = ["train1", "train2", "train3", "train4", "train5", "train6"]
EXTRA_TRAIN = ["indy_20160927_06", "indy_20160930_02", "indy_20160930_05",
               "indy_20161025_04", "indy_20161026_03", "indy_20161027_03",
               "indy_20160915_01", "indy_20160916_01", "indy_20160921_01",
               "indy_20160927_04", "indy_20161206_02", "indy_20161207_02",
               "indy_20161212_02", "indy_20161220_02", "indy_20170123_02",
               "indy_20170124_01", "indy_20170127_03", "indy_20170131_02"]

# --- headline metrics (R², untouched test1 after eval1 model selection) ---
# 'wide' single model, 24 sessions (LOG-056). The saved checkpoint scores 0.668;
# the research harness measured 0.677 -- the ~0.01 gap is training-loop variance.
METRICS = {
    "test_r2_fp32": 0.668,       # saved checkpoint (train_and_save)
    "test_r2_harness": 0.677,    # research/iter8 harness measurement
    "test_r2_ensemble3": 0.675,  # a single 'wide' model already matches the ensemble
    "params": 100_290,
    "size_kb_fp32": 392,
    "size_kb_int8": 98,
    "baseline_6session_small_r2": 0.529,   # 6-session small model (start point)
    "small_24session_r2": 0.655,           # smaller F32/H32 variant (25 kB int8)
}
