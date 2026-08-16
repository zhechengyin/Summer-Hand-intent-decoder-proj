"""Strictly causal neural inputs for the Indy 32-channel model."""

from __future__ import annotations

import numpy as np


def causal_ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    """EWMA along the final axis; output at t never reads t+1."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    output = values.astype(np.float64, copy=True)
    for index in range(1, values.shape[-1]):
        output[..., index] = (
            alpha * values[..., index] + (1 - alpha) * output[..., index - 1]
        )
    return output.astype(np.float32)


def multiscale_counts(
    counts: np.ndarray, alphas: tuple[float, ...] = (1.0, 0.1)
) -> np.ndarray:
    """Concatenate raw counts and requested causal EWMA timescales."""
    blocks = [
        counts.astype(np.float32) if alpha == 1 else causal_ewma(counts, alpha)
        for alpha in alphas
    ]
    return np.concatenate(blocks, axis=0)
