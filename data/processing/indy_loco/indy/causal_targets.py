"""Causal kinematic transforms used when building Indy processed artifacts."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def causal_sample_hold(
    timestamps: np.ndarray,
    values: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Return the latest value observed at or before each query time."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values)
    query_times = np.asarray(query_times, dtype=np.float64)
    if timestamps.ndim != 1 or query_times.ndim != 1:
        raise ValueError("timestamps and query_times must be one-dimensional")
    if values.ndim < 1 or values.shape[0] != timestamps.size:
        raise ValueError("values must use timestamps on its first axis")
    if timestamps.size == 0 or not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamps must be non-empty and strictly increasing")
    indices = np.searchsorted(timestamps, query_times, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("query_times cannot precede the first timestamp")
    return values[np.minimum(indices, timestamps.size - 1)]


def causal_velocity(
    position: np.ndarray,
    sample_period_s: float,
    lowpass_hz: float | None = 3.0,
) -> np.ndarray:
    """Filter forward and compute a backward-difference velocity."""
    position = np.asarray(position, dtype=np.float64)
    if position.ndim != 2:
        raise ValueError("position must have shape (time, coordinates)")
    if sample_period_s <= 0:
        raise ValueError("sample_period_s must be positive")

    filtered = position
    if lowpass_hz:
        nyquist = 0.5 / sample_period_s
        if not 0 < lowpass_hz < nyquist:
            raise ValueError(f"lowpass_hz must be between 0 and {nyquist:g}")
        sos = butter(4, lowpass_hz / nyquist, btype="low", output="sos")
        zi = sosfilt_zi(sos)
        filtered = np.empty_like(position)
        for axis in range(position.shape[1]):
            filtered[:, axis], _ = sosfilt(
                sos, position[:, axis], zi=zi * position[0, axis]
            )

    velocity = np.zeros_like(filtered)
    velocity[1:] = np.diff(filtered, axis=0) / sample_period_s
    return velocity.astype(np.float32)
