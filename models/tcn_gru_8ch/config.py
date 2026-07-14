"""8-channel STM32 decoder — exact spec of record (importable).

The deployable decoder: 8 spike-detection channels -> multi-timescale input
(raw + causal EWMA, 16 features) -> dilated-TCN + STRICTLY-CAUSAL (unidirectional)
GRU (`models.tcn_gru.build_net`, bidir=False), STM32-sized, trained on 24 sessions.
Causal because real-time inference has no future; int8 is lossless. Deployable
TEST R2 = 0.646 (3-seed) / ~0.61 single model; bidirectional 0.677 is OFFLINE only.
See LOG-062/068/070.

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

# Multi-timescale input (LOG-068/070): expand each of the 8 channels into raw +
# causal EWMA(alpha=0.2) = 2 scales -> 16 features. +0.028 R2 over single-scale;
# saturates at 2 scales; alpha=0.2 (tau ~160 ms) is best. Model input dim = 16.
MULTISCALE = [1.0, 0.2]

# Channel selection is FIXED to the top-8 firing electrodes of the original 6
# training sessions. Re-selecting on more data OVERFITS (0.628 -> 0.502, LOG-053);
# learned/corr selection also lose to firing-rate (LOG-043/046). Do not re-select.

# --- model hyperparameters (merged over BASE) ---
# STRICTLY CAUSAL (bidir=False): bidirectional is NOT deployable at inference (it
# needs the whole future); at 40 ms/bin even a 2-frame lookahead is 80 ms, too slow
# (LOG-062). 'wide' (F64/H64/L1) is the best causal config -- architecture,
# capacity, smoothing and more/distant data don't beat it (LOG-064/065).
MODEL = {**BASE, "dils": [1, 2, 4, 8], "H": 64, "L": 1, "F": 64, "bidir": False,
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
# DEPLOYABLE = strictly causal + multiscale input. Bidir numbers are OFFLINE only.
METRICS = {
    "test_r2_causal_multiscale_3seed": 0.646,  # DEPLOYABLE best: causal, 2-scale input, 3-seed (LOG-070)
    "test_r2_causal_single_scale": 0.606,      # causal, single-scale input (pre-multiscale)
    "test_r2_bidir_offline": 0.677,            # offline upper bound (bidirectional, NOT deployable)
    "size_kb_int8": 75,          # ~75 KB (16-feat input only grows the first conv)
    "compute_ms_per_pred": 5.6,  # causal, 1 CPU core
    "baseline_6session_small_r2": 0.529,       # start point
    # Single multiscale model ~0.61; the 0.646 uses a 3-seed ensemble.
}
