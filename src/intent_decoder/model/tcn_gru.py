"""Strictly causal TCN+GRU used by active and deployable pipelines."""
from __future__ import annotations

import numpy as np


def corr(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    actual = actual - actual.mean(0)
    predicted = predicted - predicted.mean(0)
    denominator = np.linalg.norm(actual, axis=0) * np.linalg.norm(predicted, axis=0)
    return (actual * predicted).sum(0) / np.where(denominator == 0, 1e-9, denominator)


def r2(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    residual = ((actual - predicted) ** 2).sum(0)
    total = ((actual - actual.mean(0)) ** 2).sum(0)
    return 1.0 - residual / np.where(total == 0, 1e-9, total)


def build_net(config: dict, n_channels: int):
    """Build a TCN+GRU that cannot consume future timesteps.

    Right-side convolution padding is cropped immediately, and the GRU is always
    unidirectional. A configuration requesting bidirectionality is rejected.
    """
    import torch.nn as nn

    if config.get("bidir", False):
        raise ValueError("The stable decoder forbids bidirectional=True")

    activations = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "mish": nn.Mish,
        "selu": nn.SELU,
    }
    activation = activations.get(config.get("act", "relu"), nn.ReLU)

    class CausalTCNGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            filters = config["F"]
            spatial = [nn.Conv1d(n_channels, filters, 1), nn.BatchNorm1d(filters)]
            if config.get("sp_act", True):
                spatial.append(activation())
            self.spatial = nn.Sequential(*spatial)
            self.convolutions = nn.ModuleList(
                [nn.Conv1d(filters, filters, 3, padding=2 * dilation, dilation=dilation)
                 for dilation in config["dils"]]
            )
            self.padding = [2 * dilation for dilation in config["dils"]]
            self.activation = activation()
            self.dropout = nn.Dropout(config["dropout"])
            self.gru = nn.GRU(
                filters,
                config["H"],
                config["L"],
                batch_first=True,
                bidirectional=False,
                dropout=config["dropout"] if config["L"] > 1 else 0.0,
            )
            self.head = nn.Linear(config["H"], config.get("n_out", 2))

        def forward(self, values):
            encoded = self.spatial(values)
            for convolution, padding in zip(self.convolutions, self.padding):
                # Crop the right padding: output[t] sees only input[:t+1].
                encoded = self.activation(convolution(encoded)[:, :, :-padding] + encoded)
            encoded, _ = self.gru(self.dropout(encoded).transpose(1, 2))
            return self.head(encoded)

    return CausalTCNGRU()


def causal_config(n_out: int = 2) -> dict:
    """Return current causal architecture and training defaults."""
    return {
        "F": 64,
        "H": 64,
        "L": 1,
        "dils": [1, 2, 4, 8],
        "bidir": False,
        "dropout": 0.3,
        "lr": 1e-3,
        "wd": 1e-3,
        "epochs": 60,
        "bs": 32,
        "noise": 0.1,
        "chdrop": 0.1,
        "cosine": True,
        "act": "relu",
        "n_out": n_out,
    }
