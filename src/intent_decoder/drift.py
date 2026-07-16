"""Label-free session-drift indicators."""
from __future__ import annotations

import numpy as np


def top_channel_overlap(
    observation_counts: np.ndarray, reference_channels: np.ndarray, n: int
) -> float:
    """Fraction of reference top-N channels retained in an observation prefix."""
    observed = set(np.argsort(observation_counts.mean(1))[-n:].tolist())
    reference = set(np.asarray(reference_channels).tolist())
    return len(observed & reference) / n


def prediction_std_ratio(prediction: np.ndarray, training_target_std: np.ndarray) -> float:
    """Mean predicted-output standard deviation relative to training target scale."""
    prediction = prediction.reshape(-1, prediction.shape[-1])
    return float(np.mean(prediction.std(0) / (np.asarray(training_target_std) + 1e-9)))
