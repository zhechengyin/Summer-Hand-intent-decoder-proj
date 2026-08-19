"""Streaming reference runtime for the Tiny deployment bundle."""

from __future__ import annotations

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
        raise ValueError("Tiny runtime supports auto, cpu, or cuda")
    return torch.device(requested)


class TinyRuntime:
    """60-second calibration followed by exact block-reset causal replay."""

    def __init__(self, *, device: str = "auto") -> None:
        self.device = _device(device)
        self.model, checkpoint = load_checkpoint(CHECKPOINT, map_location=self.device)
        self.model.to(self.device)
        self.channels = np.asarray(checkpoint["channels"], dtype=np.int64)
        self.std_floor = np.asarray(
            checkpoint["feature_std_floor"], dtype=np.float32
        ).reshape(64)
        self.target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        """Begin a new session cold-start."""
        self.seen_bins = 0
        self.ewma: np.ndarray | None = None
        self.feature_sum = np.zeros(64, dtype=np.float64)
        self.feature_square_sum = np.zeros(64, dtype=np.float64)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.block = np.zeros((64, WINDOW_BINS), dtype=np.float32)
        self.block_position = 0

    def _selected_counts(self, raw_counts: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_counts, dtype=np.float32).reshape(-1)
        if values.size == 32:
            selected = values
        elif values.size > int(self.channels.max()):
            selected = values[self.channels]
        else:
            raise ValueError("Expected 32 selected counts or a full recording")
        if not np.isfinite(selected).all() or np.any(selected < 0):
            raise ValueError("Counts must be finite and non-negative")
        return selected

    def _feature(self, raw_counts: np.ndarray) -> np.ndarray:
        counts = self._selected_counts(raw_counts)
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

    def _infer(self, timestep: int) -> np.ndarray:
        values = torch.from_numpy(self.block[None]).to(self.device)
        with torch.inference_mode():
            normalized = self.model(values)[0, timestep].cpu().numpy()
        return (normalized * self.target_std + self.target_mean).astype(np.float32)

    def push(self, raw_counts: np.ndarray) -> Prediction | None:
        """Consume one 40 ms count bin; return a prediction when available."""
        feature = self._feature(raw_counts)
        index = self.seen_bins
        self.seen_bins += 1

        if index < CALIBRATION_BINS:
            self.feature_sum += feature
            self.feature_square_sum += feature.astype(np.float64) ** 2
            if self.seen_bins == CALIBRATION_BINS:
                self._freeze_stats()
            return None

        if self.mean is None or self.std is None:
            raise RuntimeError("Calibration statistics were not frozen")
        if self.block_position == 0:
            self.block.fill(0.0)
        self.block[:, self.block_position] = (feature - self.mean) / self.std
        velocity = self._infer(self.block_position)
        self.block_position = (self.block_position + 1) % WINDOW_BINS
        return Prediction(index, (index + 1) * BIN_SECONDS, velocity)
