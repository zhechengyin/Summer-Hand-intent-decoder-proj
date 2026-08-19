"""Standalone architecture for the promoted Phase 6 Indy decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class PointwiseLayerNorm(nn.Module):
    """Apply layer normalization independently at every time step."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(features)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.normalization(values.transpose(1, 2)).transpose(1, 2)


class Indy96ChannelTCNGRU(nn.Module):
    """Strictly causal 96-channel, 64/64 TCN+GRU inference model.

    Input shape is ``(batch, 192, time)``: 96 raw count streams followed by
    their 96 causal-EWMA streams. Training used paired channel dropout, but
    dropout is intentionally absent from the inference graph.
    """

    physical_channels = 96
    input_features = 192
    tcn_width = 64
    gru_width = 64
    kernel_size = 3
    dilations = (1, 2, 4, 8)
    output_features = 2

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
                    self.kernel_size,
                    padding=(self.kernel_size - 1) * dilation,
                    dilation=dilation,
                )
                for dilation in self.dilations
            ]
        )
        self.padding = [
            (self.kernel_size - 1) * dilation for dilation in self.dilations
        ]
        self.activation = nn.ReLU()
        self.gru = nn.GRU(
            self.tcn_width,
            self.gru_width,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(self.gru_width, self.output_features)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.input_features:
            raise ValueError(
                "Expected input shape (batch, 192, time); "
                f"received {tuple(values.shape)}"
            )
        encoded = self.spatial(values)
        for convolution, padding in zip(
            self.convolutions, self.padding, strict=True
        ):
            convolved = convolution(encoded)
            if padding:
                convolved = convolved[:, :, :-padding]
            encoded = self.activation(convolved + encoded)
        states, _ = self.gru(encoded.transpose(1, 2))
        return self.head(states)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[Indy96ChannelTCNGRU, dict[str, Any]]:
    """Load the trusted promoted checkpoint and return an eval-mode model."""

    checkpoint = torch.load(
        Path(path), map_location=map_location, weights_only=False
    )
    expected = {
        "physical_channel_count": 96,
        "input_feature_count": 192,
        "parameter_count": 86_978,
        "channel_selection": "all96",
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Checkpoint {key}={checkpoint.get(key)!r}; expected {value!r}"
            )
    config = checkpoint.get("experiment_config", {})
    if config.get("channel_dropout") != 0.2:
        raise ValueError("Checkpoint is not the promoted Phase 6 configuration")
    if checkpoint.get("january_loaded") is not False:
        raise ValueError("Checkpoint metadata does not preserve the January lock")

    model = Indy96ChannelTCNGRU()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint
