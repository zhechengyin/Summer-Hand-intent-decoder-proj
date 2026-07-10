"""Small fused EEG temporal CNN + fNIRS feature branch.

This model is intentionally modest because ds004022 has few trials per subject.
It consumes aligned EEG epochs and fNIRS hemodynamic features:

    EEG epoch -> temporal conv -> depthwise spatial conv -> pooled embedding
    fNIRS features -> StandardScaler -> small dense embedding
    concat embeddings -> dense classifier

The estimator owns its scalers and is fit from scratch inside each CV split.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import cfg_get


class _FusedTemporalCNNNet:
    """Factory wrapper to keep torch imports local until the model is used."""

    @staticmethod
    def build(n_eeg_ch: int, n_fnirs: int, n_classes: int,
              temporal_filters: int, spatial_multiplier: int,
              separable_filters: int, kernel_size: int, dropout: float,
              fnirs_hidden: int, fusion_hidden: int):
        import torch
        import torch.nn as nn

        spatial_filters = int(temporal_filters) * int(spatial_multiplier)
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.eeg = nn.Sequential(
                    nn.Conv2d(1, temporal_filters, (1, kernel_size),
                              padding=(0, kernel_size // 2), bias=False),
                    nn.BatchNorm2d(temporal_filters),
                    nn.Conv2d(temporal_filters, spatial_filters,
                              (n_eeg_ch, 1), groups=temporal_filters,
                              bias=False),
                    nn.BatchNorm2d(spatial_filters),
                    nn.GELU(),
                    nn.AvgPool2d((1, 4)),
                    nn.Dropout(float(dropout)),
                    nn.Conv2d(spatial_filters, spatial_filters, (1, 15),
                              padding=(0, 7), groups=spatial_filters,
                              bias=False),
                    nn.Conv2d(spatial_filters, separable_filters, (1, 1),
                              bias=False),
                    nn.BatchNorm2d(separable_filters),
                    nn.GELU(),
                    nn.AvgPool2d((1, 4)),
                    nn.Dropout(float(dropout)),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                if n_fnirs > 0:
                    self.fnirs = nn.Sequential(
                        nn.Linear(n_fnirs, fnirs_hidden),
                        nn.GELU(),
                        nn.Dropout(float(dropout)),
                    )
                    head_in = separable_filters + fnirs_hidden
                else:
                    self.fnirs = None
                    head_in = separable_filters
                self.head = nn.Sequential(
                    nn.Linear(head_in, fusion_hidden),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(fusion_hidden, n_classes),
                )

            def forward(self, eeg, fnirs):
                eeg_z = self.eeg(eeg)
                if self.fnirs is not None:
                    z = torch.cat([eeg_z, self.fnirs(fnirs)], dim=1)
                else:
                    z = eeg_z
                return self.head(z)

        return Net()


class FusedTemporalCNNClassifier:
    """PyTorch classifier for aligned EEG epochs plus fNIRS features."""

    def __init__(self, eeg_decimate=5, temporal_filters=8,
                 spatial_multiplier=2, separable_filters=16, kernel_size=65,
                 fnirs_hidden=16, fusion_hidden=32, dropout=0.30, lr=1e-3,
                 weight_decay=1e-2, batch_size=16, epochs=40, patience=8,
                 validation_fraction=0.20, random_state=42, num_threads=1,
                 moving_average_window=1, verbose=False):
        self.eeg_decimate = eeg_decimate
        self.temporal_filters = temporal_filters
        self.spatial_multiplier = spatial_multiplier
        self.separable_filters = separable_filters
        self.kernel_size = kernel_size
        self.fnirs_hidden = fnirs_hidden
        self.fusion_hidden = fusion_hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.num_threads = num_threads
        self.moving_average_window = moving_average_window
        self.verbose = verbose

    def _moving_average(self, X):
        X = np.asarray(X, dtype=np.float32)
        win = int(self.moving_average_window)
        if win <= 1:
            return X
        # Causal moving average over time: y[t] = mean(x[max(0,t-win+1):t+1]).
        csum = np.cumsum(X, axis=2, dtype=np.float64)
        csum = np.concatenate([np.zeros((*X.shape[:2], 1)), csum], axis=2)
        t = np.arange(X.shape[2])
        starts = np.maximum(0, t + 1 - win)
        sums = csum[:, :, t + 1] - csum[:, :, starts]
        counts = (t + 1 - starts).astype(np.float64)
        return (sums / counts[None, None, :]).astype(np.float32)

    def _decimate(self, X):
        X = np.asarray(X, dtype=np.float32)
        step = max(1, int(self.eeg_decimate))
        return X[..., ::step] if step > 1 else X

    def _fit_transform_eeg(self, X):
        X = self._moving_average(X)
        X = self._decimate(X)
        self.eeg_mean_ = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        std = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
        self.eeg_std_ = np.maximum(std, 1e-6)
        return ((X - self.eeg_mean_) / self.eeg_std_).astype(np.float32)

    def _transform_eeg(self, X):
        X = self._moving_average(X)
        X = self._decimate(X)
        return ((X - self.eeg_mean_) / self.eeg_std_).astype(np.float32)

    def _fit_transform_fnirs(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] == 0:                       # EEG-only (no fNIRS branch)
            self.fnirs_scaler_ = None
            return X
        self.fnirs_scaler_ = StandardScaler()
        return self.fnirs_scaler_.fit_transform(X).astype(np.float32)

    def _transform_fnirs(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.fnirs_scaler_ is None:
            return X
        return self.fnirs_scaler_.transform(X).astype(np.float32)

    def _split_indices(self, y_encoded: np.ndarray):
        n = len(y_encoded)
        n_classes = len(np.unique(y_encoded))
        frac = float(self.validation_fraction)
        if frac <= 0 or n < 2 * n_classes:
            return np.arange(n), np.array([], dtype=int)
        _, counts = np.unique(y_encoded, return_counts=True)
        if counts.min() < 2:
            return np.arange(n), np.array([], dtype=int)

        val_size = max(n_classes, int(round(n * frac)))
        val_size = min(val_size, n - n_classes)
        if val_size < n_classes:
            return np.arange(n), np.array([], dtype=int)

        from sklearn.model_selection import train_test_split

        idx = np.arange(n)
        train_idx, val_idx = train_test_split(
            idx, test_size=val_size, stratify=y_encoded,
            random_state=int(self.random_state))
        return train_idx, val_idx

    def fit(self, eeg_X, fnirs_X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if int(self.num_threads) > 0:
            torch.set_num_threads(int(self.num_threads))
        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state))

        eeg = self._fit_transform_eeg(eeg_X)
        fnirs = self._fit_transform_fnirs(fnirs_X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_encoded = np.searchsorted(self.classes_, y).astype(np.int64)

        n_eeg_ch = eeg.shape[1]
        n_fnirs = fnirs.shape[1]
        model = _FusedTemporalCNNNet.build(
            n_eeg_ch=n_eeg_ch,
            n_fnirs=n_fnirs,
            n_classes=len(self.classes_),
            temporal_filters=int(self.temporal_filters),
            spatial_multiplier=int(self.spatial_multiplier),
            separable_filters=int(self.separable_filters),
            kernel_size=int(self.kernel_size),
            dropout=float(self.dropout),
            fnirs_hidden=int(self.fnirs_hidden),
            fusion_hidden=int(self.fusion_hidden),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=float(self.lr),
                                weight_decay=float(self.weight_decay))
        train_idx, val_idx = self._split_indices(y_encoded)

        def loader(indices, shuffle):
            eeg_tensor = torch.from_numpy(eeg[indices]).unsqueeze(1)
            fnirs_tensor = torch.from_numpy(fnirs[indices])
            y_tensor = torch.from_numpy(y_encoded[indices])
            ds = TensorDataset(eeg_tensor, fnirs_tensor, y_tensor)
            gen = torch.Generator()
            gen.manual_seed(int(self.random_state))
            return DataLoader(ds, batch_size=min(int(self.batch_size), len(ds)),
                              shuffle=shuffle, generator=gen if shuffle else None)

        train_loader = loader(train_idx, True)
        val_loader = loader(val_idx, False) if len(val_idx) else None
        best_state, best_loss, stale = None, float("inf"), 0

        for epoch in range(int(self.epochs)):
            model.train()
            total, count = 0.0, 0
            for xb_eeg, xb_fnirs, yb in train_loader:
                xb_eeg = xb_eeg.to(device)
                xb_fnirs = xb_fnirs.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb_eeg, xb_fnirs), yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(yb)
                count += len(yb)
            score = total / max(1, count)

            if val_loader is not None:
                model.eval()
                total, count = 0.0, 0
                with torch.no_grad():
                    for xb_eeg, xb_fnirs, yb in val_loader:
                        xb_eeg = xb_eeg.to(device)
                        xb_fnirs = xb_fnirs.to(device)
                        yb = yb.to(device)
                        loss = loss_fn(model(xb_eeg, xb_fnirs), yb)
                        total += float(loss.item()) * len(yb)
                        count += len(yb)
                score = total / max(1, count)

            if score < best_loss - 1e-5:
                best_loss = score
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if int(self.patience) > 0 and stale >= int(self.patience):
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model.cpu().eval()
        self.n_iter_ = epoch + 1
        self.loss_ = float(best_loss)
        return self

    def predict_proba(self, eeg_X, fnirs_X):
        import torch

        eeg = self._transform_eeg(eeg_X)
        fnirs = self._transform_fnirs(fnirs_X)
        with torch.no_grad():
            eeg_tensor = torch.from_numpy(eeg).unsqueeze(1)
            fnirs_tensor = torch.from_numpy(fnirs)
            logits = self.model_(eeg_tensor, fnirs_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict(self, eeg_X, fnirs_X):
        return self.classes_[np.argmax(self.predict_proba(eeg_X, fnirs_X),
                                       axis=1)]


def build_temporal_cnn_classifier(
        cfg: dict, prefix: str = "temporal_cnn") -> FusedTemporalCNNClassifier:
    """Build the fused temporal CNN classifier from config."""
    seed = int(cfg_get(cfg, "seed", 42))
    return FusedTemporalCNNClassifier(
        eeg_decimate=int(cfg_get(cfg, f"{prefix}.eeg_decimate", 5)),
        temporal_filters=int(cfg_get(cfg, f"{prefix}.temporal_filters", 8)),
        spatial_multiplier=int(cfg_get(cfg, f"{prefix}.spatial_multiplier", 2)),
        separable_filters=int(cfg_get(cfg, f"{prefix}.separable_filters", 16)),
        kernel_size=int(cfg_get(cfg, f"{prefix}.kernel_size", 65)),
        fnirs_hidden=int(cfg_get(cfg, f"{prefix}.fnirs_hidden", 16)),
        fusion_hidden=int(cfg_get(cfg, f"{prefix}.fusion_hidden", 32)),
        dropout=float(cfg_get(cfg, f"{prefix}.dropout", 0.30)),
        lr=float(cfg_get(cfg, f"{prefix}.lr", 1e-3)),
        weight_decay=float(cfg_get(cfg, f"{prefix}.weight_decay", 1e-2)),
        batch_size=int(cfg_get(cfg, f"{prefix}.batch_size", 16)),
        epochs=int(cfg_get(cfg, f"{prefix}.epochs", 40)),
        patience=int(cfg_get(cfg, f"{prefix}.patience", 8)),
        validation_fraction=float(cfg_get(
            cfg, f"{prefix}.validation_fraction", 0.20)),
        random_state=seed,
        num_threads=int(cfg_get(cfg, f"{prefix}.num_threads", 1)),
        moving_average_window=int(cfg_get(
            cfg, f"{prefix}.moving_average_window", 1)),
        verbose=bool(cfg_get(cfg, f"{prefix}.verbose", False)),
    )
