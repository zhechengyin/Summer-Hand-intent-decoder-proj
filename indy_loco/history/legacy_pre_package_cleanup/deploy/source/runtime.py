"""Phase 9 rolling past-window runtime for the Midsize deployment bundle."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from .model import load_checkpoint
except ImportError:  # Allow direct execution/import from this directory.
    from model import load_checkpoint

CHECKPOINT = Path(__file__).with_name("checkpoint.pt")
CALIBRATION_BINS = 1_500
WINDOW_BINS = 50
BIN_SECONDS = 0.04
EWMA_ALPHA = 0.1


@dataclass(frozen=True)
class Prediction:
    """One velocity prediction at a known bin end."""

    bin_index: int
    bin_end_time_s: float
    velocity_xy: np.ndarray


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("Midsize runtime supports auto, cpu, or cuda")
    return torch.device(requested)


class MidsizeRuntime:
    """60-second calibration followed by a continuous 50-bin past window."""

    def __init__(self, *, device: str = "auto") -> None:
        self.device = _device(device)
        self.model, checkpoint = load_checkpoint(CHECKPOINT, map_location=self.device)
        self.model.to(self.device)
        self.std_floor = np.asarray(
            checkpoint["feature_std_floor"], dtype=np.float32
        ).reshape(192)
        self.target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        """Begin a new session cold-start."""
        self.seen_bins = 0
        self.ewma: np.ndarray | None = None
        self.feature_sum = np.zeros(192, dtype=np.float64)
        self.feature_square_sum = np.zeros(192, dtype=np.float64)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.window: deque[np.ndarray] = deque(maxlen=WINDOW_BINS)

    @staticmethod
    def _counts(raw_counts: np.ndarray) -> np.ndarray:
        counts = np.asarray(raw_counts, dtype=np.float32).reshape(-1)
        if counts.size != 96:
            raise ValueError("Midsize runtime requires channels 0..95 in order")
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("Counts must be finite and non-negative")
        return counts

    def _feature(self, raw_counts: np.ndarray) -> np.ndarray:
        counts = self._counts(raw_counts)
        self.ewma = (
            counts.copy()
            if self.ewma is None
            else EWMA_ALPHA * counts + (1.0 - EWMA_ALPHA) * self.ewma
        )
        return np.concatenate((counts, self.ewma)).astype(np.float32)

    def _freeze_stats(self) -> None:
        self.mean = (self.feature_sum / CALIBRATION_BINS).astype(np.float32)
        variance = self.feature_square_sum / CALIBRATION_BINS - self.mean**2
        local_std = np.sqrt(np.maximum(variance, 0.0)) + 1e-6
        self.std = np.maximum(local_std, self.std_floor).astype(np.float32)

    def _infer(self) -> np.ndarray:
        if self.mean is None or self.std is None or len(self.window) != WINDOW_BINS:
            raise RuntimeError("A complete calibrated past window is required")
        raw_window = np.stack(self.window, axis=1)
        normalized = ((raw_window - self.mean[:, None]) / self.std[:, None]).astype(
            np.float32
        )
        values = torch.from_numpy(normalized[None]).to(self.device)
        with torch.inference_mode():
            output = self.model(values)[0, -1].cpu().numpy()
        return (output * self.target_std + self.target_mean).astype(np.float32)

    def push(self, raw_counts: np.ndarray) -> Prediction | None:
        """Consume one 40 ms count bin; return a prediction when available."""
        feature = self._feature(raw_counts)
        index = self.seen_bins
        self.seen_bins += 1
        self.window.append(feature)

        if index < CALIBRATION_BINS:
            self.feature_sum += feature
            self.feature_square_sum += feature.astype(np.float64) ** 2
            if self.seen_bins < CALIBRATION_BINS:
                return None
            self._freeze_stats()
        velocity = self._infer()
        return Prediction(index, (index + 1) * BIN_SECONDS, velocity)
