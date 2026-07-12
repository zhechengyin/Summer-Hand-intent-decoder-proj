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
MODEL = {**BASE, "dils": [1, 2, 4, 8], "H": 32, "L": 1, "F": 32,
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
METRICS = {
    "test_r2_fp32": 0.655,       # 24-session single model, 100 kB (LOG-055)
    "test_r2_int8": 0.655,       # ~27 kB, lossless quantization (LOG-054)
    "test_r2_ensemble3": 0.675,  # 3-seed ensemble, 24 sessions (3x cost)
    "test_r": 0.812,
    "params": 25_570,
    "size_kb_fp32": 100,
    "size_kb_int8": 27,
    "baseline_6session_r2": 0.529,   # what more data improved on
}
