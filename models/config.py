"""Current best model — exact spec of record (importable).

The decoder of record: per-electrode multiunit spike rates -> 2D fingertip
velocity, via a dilated causal TCN + bidirectional GRU (models.best_model.build_net).
This module is the single source of truth for the winning configuration; the
research scripts (crosssession.py etc.) define the same values inline.
"""
from models.best_model import BASE

# --- preprocessing (per-electrode pipeline) ---
BIN_S = 0.04                 # 40 ms bins -> 25 Hz
WINDOW_S = 2.0               # 2 s sequence windows
VEL_LOWPASS_HZ = 3.0         # low-pass finger position before differentiating (LOG-030)
RATE_SMOOTH_SIGMA_BINS = 1.0 # Gaussian smoothing of firing rates (LOG-032)
N_ELECTRODES = 96            # indy M1 array; per-electrode multiunit (session-consistent)
N_OUT = 2                    # 2D: top-2 velocity-variance movement axes

# --- model hyperparameters (merged over core.BASE) ---
MODEL = {**BASE, "dils": [1, 2, 4, 8, 16], "H": 64, "L": 2, "F": 64,
         "epochs": 60, "noise": 0.1, "chdrop": 0.1, "cosine": True,
         "act": "relu", "n_out": N_OUT}

# --- headline metrics (Pearson r, held-out sessions never seen in training) ---
METRICS = {
    "held_out_cross_session_96ch": 0.87,   # train 6 indy sessions, test 2 held-out (LOG-033)
    "held_out_8ch_firing_top8": 0.76,      # hardware constraint, top-8 by firing (LOG-042)
    "params": 192_000,
    "size_mb": 0.77,
    "inference_ms_bidirectional": 6.0,     # per prediction, 1 CPU core
    "inference_ms_causal": 3.7,
}
