"""Standalone 32-channel 48/48 causal TCN+GRU deployment model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class PointwiseLayerNorm(nn.Module):
    """Apply layer normalization independently at each time step."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(features)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.normalization(values.transpose(1, 2)).transpose(1, 2)


class TinyTCNGRU(nn.Module):
    """Four-block width-48 TCN followed by a width-48 unidirectional GRU."""

    input_features = 64
    tcn_width = 48
    gru_width = 48
    dilations = (1, 2, 4, 8)

    def __init__(self) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv1d(self.input_features, self.tcn_width, 1),
            PointwiseLayerNorm(self.tcn_width),
            nn.ReLU(),
        )
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    self.tcn_width,
                    self.tcn_width,
                    3,
                    padding=2 * dilation,
                    dilation=dilation,
                )
                for dilation in self.dilations
            ]
        )
        self.padding = [2 * dilation for dilation in self.dilations]
        self.activation = nn.ReLU()
        self.gru = nn.GRU(
            self.tcn_width,
            self.gru_width,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(self.gru_width, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.input_features:
            raise ValueError(
                "Expected input shape (batch, 64, time); "
                f"received {tuple(values.shape)}"
            )
        encoded = self.spatial(values)
        for convolution, padding in zip(self.convolutions, self.padding, strict=True):
            encoded = self.activation(convolution(encoded)[:, :, :-padding] + encoded)
        states, _ = self.gru(encoded.transpose(1, 2))
        return self.head(states)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TinyTCNGRU, dict[str, Any]]:
    """Load and validate the frozen Tiny deployment checkpoint."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if checkpoint.get("parameter_count") != 45_266:
        raise ValueError("Checkpoint is not the retained Tiny model")
    config = checkpoint.get("config", {})
    expected = {"F": 48, "H": 48, "L": 1, "bidir": False, "n_out": 2}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected Tiny config {key}={config.get(key)!r}")
    channels = checkpoint.get("channels", [])
    if len(channels) != 32 or len(set(channels)) != 32:
        raise ValueError("Tiny checkpoint must contain 32 unique channels")

    model = TinyTCNGRU()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint
