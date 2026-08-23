"""PyTorch fully connected classifier for spectral feature vectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class AnnTrainingResult:
    best_epoch: int
    best_validation_loss: float
    epochs_run: int


class MstAnnNetwork(nn.Module):
    """Exact requested architecture: input -> FC(64) -> ReLU -> FC(2)."""

    def __init__(self, input_features: int):
        super().__init__()
        if input_features <= 0:
            raise ValueError("input_features must be positive.")
        self.fc1 = nn.Linear(input_features, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(values)))


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return torch.device(requested)


class MstAnnClassifier:
    """Scikit-like wrapper around the two-layer PyTorch network."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 300,
        validation_fraction: float = 0.2,
        patience: int = 30,
        random_state: int = 2020,
        device: str = "auto",
    ):
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("Invalid learning rate or weight decay.")
        if batch_size <= 0 or epochs <= 0 or patience <= 0:
            raise ValueError("Batch size, epochs, and patience must be positive.")
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one.")
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.device = choose_device(device)
        self.training_result: AnnTrainingResult | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MstAnnClassifier":
        values = np.asarray(x, dtype=np.float32)
        labels = np.asarray(y).reshape(-1)
        if values.ndim != 2 or len(values) != len(labels):
            raise ValueError("Expected matching 2-D features and 1-D labels.")
        if not np.isfinite(values).all():
            raise ValueError("ANN features contain non-finite values.")
        self.classes_ = np.unique(labels)
        if self.classes_.shape != (2,):
            raise ValueError("ANN requires exactly two classes.")
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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.network = MstAnnNetwork(values.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(self.random_state)
        train_dataset = TensorDataset(
            torch.from_numpy(normalized[train_idx]),
            torch.from_numpy(encoded[train_idx]),
        )
        train_loader = DataLoader(
            train_dataset,
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
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.network(batch_x), batch_y)
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
            raise RuntimeError("ANN training did not produce a checkpoint.")
        self.network.load_state_dict(best_state)
        self.network.eval()
        self.training_result = AnnTrainingResult(best_epoch, best_loss, epoch)
        return self

    @property
    def fc1_weight_shape(self) -> tuple[int, int]:
        output_features, input_features = self.network.fc1.weight.shape
        return int(input_features), int(output_features)

    @property
    def fc2_weight_shape(self) -> tuple[int, int]:
        output_features, input_features = self.network.fc2.weight.shape
        return int(input_features), int(output_features)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        normalized = (values - self.feature_mean) / self.feature_scale
        tensor = torch.from_numpy(normalized).to(self.device)
        self.network.eval()
        with torch.inference_mode():
            probabilities = torch.softmax(self.network(tensor), dim=1)
        return probabilities.cpu().numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "format_version": "mst_ann_fc64_fc2_v2",
                "input_features": self.network.fc1.in_features,
                "classes": self.classes_,
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
                "model_state": {
                    name: value.detach().cpu()
                    for name, value in self.network.state_dict().items()
                },
                "architecture": {
                    "fc1_weight_shape_input_output": self.fc1_weight_shape,
                    "activation": "relu",
                    "fc2_weight_shape_input_output": self.fc2_weight_shape,
                },
            },
            Path(path),
        )
