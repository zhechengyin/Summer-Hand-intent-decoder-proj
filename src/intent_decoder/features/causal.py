"""Strictly past-and-present neural feature transforms."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def causal_sample_hold(
    timestamps: np.ndarray,
    values: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Return the latest value observed at or before each query time.

    Unlike linear interpolation, this operation never reads the sample after a
    query. ``values`` uses time on its first axis.
    """
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


def causal_ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    """EWMA along the final axis; output at ``t`` never reads ``t+1``."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    out = values.astype(np.float64, copy=True)
    for index in range(1, values.shape[-1]):
        out[..., index] = alpha * values[..., index] + (1 - alpha) * out[..., index - 1]
    return out.astype(np.float32)


def multiscale_counts(counts: np.ndarray, alphas: tuple[float, ...] = (1.0, 0.1)) -> np.ndarray:
    """Concatenate raw counts and requested causal EWMA timescales."""
    blocks = [counts.astype(np.float32) if alpha == 1 else causal_ewma(counts, alpha)
              for alpha in alphas]
    return np.concatenate(blocks, axis=0)


def causal_velocity(
    position: np.ndarray,
    sample_period_s: float,
    lowpass_hz: float | None = 3.0,
) -> np.ndarray:
    """Compute velocity using only position samples available at each time.

    The optional Butterworth filter runs forward once with an initial state based
    on the first sample. Velocity then uses a backward difference. Unlike
    ``sosfiltfilt`` and ``numpy.gradient``, neither operation reads ``t+1``.
    """
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
