"""Small calibration primitives shared by experiments and deployment."""
from __future__ import annotations

import numpy as np


def fit_affine(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit an independent gain and offset for each output axis."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have matching (samples, axes) shapes")
    gain = np.zeros(prediction.shape[1], dtype=np.float64)
    offset = np.zeros(prediction.shape[1], dtype=np.float64)
    for axis in range(prediction.shape[1]):
        design = np.column_stack([prediction[:, axis], np.ones(len(prediction))])
        gain[axis], offset[axis] = np.linalg.lstsq(design, target[:, axis], rcond=None)[0]
    return gain, offset


def apply_affine(prediction: np.ndarray, gain: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return prediction * gain + offset
