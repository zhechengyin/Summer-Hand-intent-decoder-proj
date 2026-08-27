"""Standalone 96-channel 64/64 causal TCN+GRU deployment model."""

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


class MidsizeTCNGRU(nn.Module):
    """Four-block width-64 TCN followed by a width-64 unidirectional GRU."""

    input_features = 192
    tcn_width = 64
    gru_width = 64
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
                "Expected input shape (batch, 192, time); "
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
) -> tuple[MidsizeTCNGRU, dict[str, Any]]:
    """Load and validate one canonical Phase-13 per-session fold checkpoint."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    expected = {
        "physical_channel_count": 96,
        "input_feature_count": 192,
        "parameter_count": 86_978,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Unexpected Midsize checkpoint {key}={checkpoint.get(key)!r}"
            )
    channels = checkpoint.get("selected_channel_indices", [])
    source_count = int(checkpoint.get("source_channel_count", 0))
    if len(channels) != 96 or len(set(channels)) != 96:
        raise ValueError("Checkpoint must select exactly 96 unique physical channels")
    if min(channels, default=-1) < 0 or max(channels, default=source_count) >= source_count:
        raise ValueError("Checkpoint channel selection is outside the source array")
    if checkpoint.get("selection_policy") != "minimum_validation_loss_test_opened_once":
        raise ValueError("Checkpoint was not selected by validation loss")
    if checkpoint.get("test_evaluated_during_training") is not False:
        raise ValueError("Checkpoint evaluated test targets during training")
    deployment = checkpoint.get("deployment_policy", {})
    if deployment.get("calibration_bins") != 10_500:
        raise ValueError("Checkpoint does not use the final seven-minute calibration contract")

    model = MidsizeTCNGRU()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint
