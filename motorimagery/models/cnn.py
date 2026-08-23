"""Tiny PyTorch CNN over channel-by-spectral-feature maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .ann import choose_device


class TinyCnnNetwork(nn.Module):
    """One spectral convolution, global pooling, and a two-class head."""

    def __init__(self, input_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(8, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.relu(self.conv(values))
        return self.classifier(self.pool(encoded).squeeze(-1))


@dataclass
class CnnTrainingResult:
    best_epoch: int
    best_validation_loss: float
    epochs_run: int


class TinyCnnClassifier:
    """Scikit-like TRAIN-only trainer for :class:`TinyCnnNetwork`."""

    def __init__(
        self,
        input_channels: int,
        features_per_channel: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 300,
        validation_fraction: float = 0.2,
        patience: int = 30,
        random_state: int = 2020,
        device: str = "auto",
    ):
        if input_channels <= 0 or features_per_channel < 3:
            raise ValueError("CNN needs positive channels and at least 3 features.")
        self.input_channels = input_channels
        self.features_per_channel = features_per_channel
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.device = choose_device(device)
        self.training_result: CnnTrainingResult | None = None

    def _reshape(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        expected = self.input_channels * self.features_per_channel
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError(f"Expected (*, {expected}) flattened CNN features.")
        return values.reshape(-1, self.input_channels, self.features_per_channel)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TinyCnnClassifier":
        values = self._reshape(x)
        labels = np.asarray(y).reshape(-1)
        if len(values) != len(labels) or not np.isfinite(values).all():
            raise ValueError("CNN features and labels are invalid.")
        self.classes_ = np.unique(labels)
        if self.classes_.shape != (2,):
            raise ValueError("CNN requires exactly two classes.")
        encoded = np.searchsorted(self.classes_, labels).astype(np.int64)
        indices = np.arange(len(values))
        train_idx, validation_idx = train_test_split(
            indices,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=encoded,
        )

        self.feature_mean = values[train_idx].mean(axis=0)
        self.feature_scale = values[train_idx].std(axis=0)
        self.feature_scale[self.feature_scale < 1e-8] = 1.0
        normalized = (values - self.feature_mean) / self.feature_scale
        torch.manual_seed(self.random_state)
        self.network = TinyCnnNetwork(self.input_channels).to(self.device)
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(normalized[train_idx]),
                torch.from_numpy(encoded[train_idx]),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        validation_x = torch.from_numpy(normalized[validation_idx]).to(self.device)
        validation_y = torch.from_numpy(encoded[validation_idx]).to(self.device)
        best_loss = float("inf")
        best_state = None
        best_epoch = 0
        stale_epochs = 0
        for epoch in range(1, self.epochs + 1):
            self.network.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(
                    self.network(batch_x.to(self.device)),
                    batch_y.to(self.device),
                )
                loss.backward()
                optimizer.step()
            self.network.eval()
            with torch.inference_mode():
                validation_loss = float(
                    criterion(self.network(validation_x), validation_y).item()
                )
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.network.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("CNN training did not produce a checkpoint.")
        self.network.load_state_dict(best_state)
        self.network.eval()
        self.training_result = CnnTrainingResult(best_epoch, best_loss, epoch)
        return self

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.network.parameters())

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = self._reshape(x)
        normalized = (values - self.feature_mean) / self.feature_scale
        tensor = torch.from_numpy(normalized).to(self.device)
        self.network.eval()
        with torch.inference_mode():
            return torch.softmax(self.network(tensor), dim=1).cpu().numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "format_version": "tiny_spectral_cnn_v1",
                "input_channels": self.input_channels,
                "features_per_channel": self.features_per_channel,
                "classes": self.classes_,
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
                "model_state": {
                    name: value.detach().cpu()
                    for name, value in self.network.state_dict().items()
                },
            },
            Path(path),
        )

