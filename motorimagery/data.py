"""Dataset loading and paper-described downsampling."""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly


def _load_mat_or_mat_gz(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            raw = f.read()
        return loadmat(io.BytesIO(raw))
    return loadmat(path)


def load_competition_train(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load BCI Competition III Dataset I training data.

    Official format:
        X: trials x electrode_channels x samples
        Y: 278 labels, values -1/+1
    """
    mat = _load_mat_or_mat_gz(path)
    if "X" not in mat or "Y" not in mat:
        raise KeyError("Training MAT file must contain variables 'X' and 'Y'.")

    x = np.asarray(mat["X"], dtype=np.float64)
    y = np.asarray(mat["Y"]).reshape(-1).astype(np.int64)

    if x.ndim != 3:
        raise ValueError(f"Expected X to be 3-D [trials, channels, samples], got {x.shape}.")
    if x.shape[0] != y.size:
        raise ValueError(f"Trial/label mismatch: X has {x.shape[0]} trials, Y has {y.size}.")
    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError(f"Expected labels -1/+1, got {np.unique(y)}.")

    return x, y


def load_competition_test(path: str | Path) -> np.ndarray:
    """Load the official unlabeled test data."""
    mat = _load_mat_or_mat_gz(path)
    if "X" not in mat:
        raise KeyError("Test MAT file must contain variable 'X'.")

    x = np.asarray(mat["X"], dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected X to be 3-D [trials, channels, samples], got {x.shape}.")
    return x


def load_true_test_labels(path: str | Path) -> np.ndarray:
    """Load the 100 true labels released by the competition organizers."""
    y = np.loadtxt(path, dtype=np.float64).reshape(-1)
    y = np.rint(y).astype(np.int64)
    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError(f"Expected labels -1/+1, got {np.unique(y)}.")
    return y


def downsample_ecog(x: np.ndarray, original_fs: int = 1000, target_fs: int = 100) -> np.ndarray:
    """Downsample ECoG epochs along the time axis.

    Xu et al. state only that 1000-Hz data are downsampled to 100 Hz.
    They do not state the exact resampling filter.

    This reproduction uses scipy.signal.resample_poly, which applies
    anti-alias filtering before decimation.
    """
    if original_fs % target_fs != 0:
        raise ValueError("This implementation expects integer downsampling ratio.")
    down = original_fs // target_fs
    return resample_poly(x, up=1, down=down, axis=-1)


def validate_dataset_shape(x: np.ndarray, expected_channels: int = 64, expected_samples: int | None = None) -> None:
    if x.ndim != 3:
        raise ValueError(f"Expected [trials, channels, samples], got {x.shape}.")
    if x.shape[1] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {x.shape[1]}.")
    if expected_samples is not None and x.shape[2] != expected_samples:
        raise ValueError(f"Expected {expected_samples} time samples, got {x.shape[2]}.")
