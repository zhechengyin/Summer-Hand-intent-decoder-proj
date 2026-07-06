"""Bare fused spiking neural network for EEG + fNIRS comparison.

This is intentionally minimal: a current-based leaky-integrate-and-fire hidden
layer with a linear readout. Raw EEG is treated as the time-varying signal, while
aligned fNIRS hemodynamic features are repeated at each SNN time step.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import cfg_get


class _SpikeFn:
    """Surrogate-gradient binary spike function."""

    @staticmethod
    def make(slope: float):
        import torch

        class Spike(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return (x > 0).to(x.dtype)

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                scale = float(slope)
                grad = 1.0 / (scale * x.abs() + 1.0) ** 2
                return grad_output * grad

        return Spike.apply


class _BareSNNNet:
    """Factory wrapper to keep torch imports local until the model is used."""

    @staticmethod
    def build(n_eeg_ch: int, n_fnirs: int, n_classes: int, hidden_size: int,
              beta: float, threshold: float, surrogate_slope: float,
              dropout: float):
        import torch
        import torch.nn as nn

        spike_fn = _SpikeFn.make(float(surrogate_slope))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj = nn.Linear(n_eeg_ch + n_fnirs, hidden_size)
                self.readout = nn.Linear(hidden_size, n_classes)
                self.drop = nn.Dropout(float(dropout))

            def forward(self, eeg, fnirs):
                # eeg: (batch, time, channels), fnirs: (batch, features)
                batch, n_time, _ = eeg.shape
                mem = torch.zeros(batch, hidden_size, device=eeg.device,
                                  dtype=eeg.dtype)
                logits = torch.zeros(batch, n_classes, device=eeg.device,
                                     dtype=eeg.dtype)
                fnirs_seq = fnirs.unsqueeze(1).expand(-1, n_time, -1)
                x_seq = torch.cat([eeg, fnirs_seq], dim=2)

                for t in range(n_time):
                    cur = self.in_proj(x_seq[:, t, :])
                    mem = float(beta) * mem + cur
                    spk = spike_fn(mem - float(threshold))
                    mem = mem * (1.0 - spk.detach())
                    if self.training:
                        spk = self.drop(spk)
                    logits = logits + self.readout(spk)
                return logits / max(1, n_time)

        return Net()


class BareSNNClassifier:
    """Sklearn-like PyTorch classifier for aligned EEG epochs + fNIRS features."""

    def __init__(self, eeg_decimate=5, time_bins=128, hidden_size=64,
                 beta=0.90, threshold=1.0, surrogate_slope=25.0,
                 dropout=0.10, lr=1e-3, weight_decay=1e-2, batch_size=16,
                 epochs=40, patience=8, validation_fraction=0.20,
                 random_state=42, num_threads=1, verbose=False):
        self.eeg_decimate = eeg_decimate
        self.time_bins = time_bins
        self.hidden_size = hidden_size
        self.beta = beta
        self.threshold = threshold
        self.surrogate_slope = surrogate_slope
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.num_threads = num_threads
        self.verbose = verbose

    def _decimate(self, X):
        X = np.asarray(X, dtype=np.float32)
        step = max(1, int(self.eeg_decimate))
        return X[..., ::step] if step > 1 else X

    def _make_time_bins(self, n_time: int):
        target = int(self.time_bins)
        if target <= 0 or n_time <= target:
            return [np.array([i]) for i in range(n_time)]
        return [idx for idx in np.array_split(np.arange(n_time), target)
                if len(idx)]

    def _apply_time_bins(self, X):
        return np.stack([X[:, :, idx].mean(axis=2) for idx in self.time_bins_],
                        axis=2).astype(np.float32)

    def _fit_transform_eeg(self, X):
        X = self._decimate(X)
        self.time_bins_ = self._make_time_bins(X.shape[2])
        X = self._apply_time_bins(X)
        self.eeg_mean_ = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        std = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
        self.eeg_std_ = np.maximum(std, 1e-6)
        X = ((X - self.eeg_mean_) / self.eeg_std_).astype(np.float32)
        return np.transpose(X, (0, 2, 1)).copy()

    def _transform_eeg(self, X):
        X = self._decimate(X)
        X = self._apply_time_bins(X)
        X = ((X - self.eeg_mean_) / self.eeg_std_).astype(np.float32)
        return np.transpose(X, (0, 2, 1)).copy()

    def _fit_transform_fnirs(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.fnirs_scaler_ = StandardScaler()
        return self.fnirs_scaler_.fit_transform(X).astype(np.float32)

    def _transform_fnirs(self, X):
        X = np.asarray(X, dtype=np.float32)
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

        model = _BareSNNNet.build(
            n_eeg_ch=eeg.shape[2],
            n_fnirs=fnirs.shape[1],
            n_classes=len(self.classes_),
            hidden_size=int(self.hidden_size),
            beta=float(self.beta),
            threshold=float(self.threshold),
            surrogate_slope=float(self.surrogate_slope),
            dropout=float(self.dropout),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=float(self.lr),
                                weight_decay=float(self.weight_decay))
        train_idx, val_idx = self._split_indices(y_encoded)

        def loader(indices, shuffle):
            ds = TensorDataset(torch.from_numpy(eeg[indices]),
                               torch.from_numpy(fnirs[indices]),
                               torch.from_numpy(y_encoded[indices]))
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
            logits = self.model_(torch.from_numpy(eeg), torch.from_numpy(fnirs))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict(self, eeg_X, fnirs_X):
        return self.classes_[np.argmax(self.predict_proba(eeg_X, fnirs_X),
                                       axis=1)]


def build_snn_classifier(cfg: dict) -> BareSNNClassifier:
    """Build the bare fused SNN classifier from config."""
    seed = int(cfg_get(cfg, "seed", 42))
    return BareSNNClassifier(
        eeg_decimate=int(cfg_get(cfg, "snn.eeg_decimate", 5)),
        time_bins=int(cfg_get(cfg, "snn.time_bins", 128)),
        hidden_size=int(cfg_get(cfg, "snn.hidden_size", 64)),
        beta=float(cfg_get(cfg, "snn.beta", 0.90)),
        threshold=float(cfg_get(cfg, "snn.threshold", 1.0)),
        surrogate_slope=float(cfg_get(cfg, "snn.surrogate_slope", 25.0)),
        dropout=float(cfg_get(cfg, "snn.dropout", 0.10)),
        lr=float(cfg_get(cfg, "snn.lr", 1e-3)),
        weight_decay=float(cfg_get(cfg, "snn.weight_decay", 1e-2)),
        batch_size=int(cfg_get(cfg, "snn.batch_size", 16)),
        epochs=int(cfg_get(cfg, "snn.epochs", 40)),
        patience=int(cfg_get(cfg, "snn.patience", 8)),
        validation_fraction=float(cfg_get(cfg, "snn.validation_fraction", 0.20)),
        random_state=seed,
        num_threads=int(cfg_get(cfg, "snn.num_threads", 1)),
        verbose=bool(cfg_get(cfg, "snn.verbose", False)),
    )


class _FeatureSNNNet:
    """LIF SNN over static or sequential feature vectors."""

    @staticmethod
    def build(n_features: int, n_classes: int, hidden_size: int, beta: float,
              threshold: float, surrogate_slope: float, dropout: float):
        import torch
        import torch.nn as nn

        spike_fn = _SpikeFn.make(float(surrogate_slope))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj = nn.Linear(n_features, hidden_size)
                self.readout = nn.Linear(hidden_size, n_classes)
                self.drop = nn.Dropout(float(dropout))

            def forward(self, x_seq):
                # x_seq: (batch, time, features)
                batch, n_time, _ = x_seq.shape
                mem = torch.zeros(batch, hidden_size, device=x_seq.device,
                                  dtype=x_seq.dtype)
                logits = torch.zeros(batch, n_classes, device=x_seq.device,
                                     dtype=x_seq.dtype)
                for t in range(n_time):
                    cur = self.in_proj(x_seq[:, t, :])
                    mem = float(beta) * mem + cur
                    spk = spike_fn(mem - float(threshold))
                    mem = mem * (1.0 - spk.detach())
                    if self.training:
                        spk = self.drop(spk)
                    logits = logits + self.readout(spk)
                return logits / max(1, n_time)

        return Net()


class FeatureSNNClassifier:
    """SNN classifier for 2-D static or 3-D sequential feature arrays."""

    def __init__(self, time_steps=32, hidden_size=64, beta=0.90,
                 threshold=1.0, surrogate_slope=25.0, dropout=0.10,
                 lr=1e-3, weight_decay=1e-2, batch_size=16, epochs=40,
                 patience=8, validation_fraction=0.20, random_state=42,
                 num_threads=1, verbose=False):
        self.time_steps = time_steps
        self.hidden_size = hidden_size
        self.beta = beta
        self.threshold = threshold
        self.surrogate_slope = surrogate_slope
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.num_threads = num_threads
        self.verbose = verbose

    def _fit_transform_X(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.input_ndim_ = X.ndim
        self.scaler_ = StandardScaler()
        if X.ndim == 2:
            Xs = self.scaler_.fit_transform(X).astype(np.float32)
            steps = max(1, int(self.time_steps))
            return np.repeat(Xs[:, None, :], steps, axis=1)
        if X.ndim == 3:
            n, t, f = X.shape
            flat = X.reshape(n * t, f)
            Xs = self.scaler_.fit_transform(flat).reshape(n, t, f)
            return Xs.astype(np.float32)
        raise ValueError("FeatureSNNClassifier expects 2-D or 3-D X")

    def _transform_X(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != self.input_ndim_:
            raise ValueError("feature rank changed between fit and predict")
        if X.ndim == 2:
            Xs = self.scaler_.transform(X).astype(np.float32)
            steps = max(1, int(self.time_steps))
            return np.repeat(Xs[:, None, :], steps, axis=1)
        n, t, f = X.shape
        Xs = self.scaler_.transform(X.reshape(n * t, f)).reshape(n, t, f)
        return Xs.astype(np.float32)

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

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if int(self.num_threads) > 0:
            torch.set_num_threads(int(self.num_threads))
        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state))

        Xs = self._fit_transform_X(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_encoded = np.searchsorted(self.classes_, y).astype(np.int64)

        model = _FeatureSNNNet.build(
            n_features=Xs.shape[2],
            n_classes=len(self.classes_),
            hidden_size=int(self.hidden_size),
            beta=float(self.beta),
            threshold=float(self.threshold),
            surrogate_slope=float(self.surrogate_slope),
            dropout=float(self.dropout),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=float(self.lr),
                                weight_decay=float(self.weight_decay))
        train_idx, val_idx = self._split_indices(y_encoded)

        def loader(indices, shuffle):
            ds = TensorDataset(torch.from_numpy(Xs[indices]),
                               torch.from_numpy(y_encoded[indices]))
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
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(yb)
                count += len(yb)
            score = total / max(1, count)

            if val_loader is not None:
                model.eval()
                total, count = 0.0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        loss = loss_fn(model(xb), yb)
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

    def predict_proba(self, X):
        import torch

        Xs = self._transform_X(X)
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(Xs))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_feature_snn_classifier(cfg: dict,
                                 prefix: str = "riemannian_snn"
                                 ) -> FeatureSNNClassifier:
    """Build an SNN classifier for static or sequential feature arrays."""
    seed = int(cfg_get(cfg, "seed", 42))
    return FeatureSNNClassifier(
        time_steps=int(cfg_get(cfg, f"{prefix}.time_steps", 32)),
        hidden_size=int(cfg_get(cfg, f"{prefix}.hidden_size", 64)),
        beta=float(cfg_get(cfg, f"{prefix}.beta", 0.90)),
        threshold=float(cfg_get(cfg, f"{prefix}.threshold", 1.0)),
        surrogate_slope=float(cfg_get(cfg, f"{prefix}.surrogate_slope", 25.0)),
        dropout=float(cfg_get(cfg, f"{prefix}.dropout", 0.10)),
        lr=float(cfg_get(cfg, f"{prefix}.lr", 1e-3)),
        weight_decay=float(cfg_get(cfg, f"{prefix}.weight_decay", 1e-2)),
        batch_size=int(cfg_get(cfg, f"{prefix}.batch_size", 16)),
        epochs=int(cfg_get(cfg, f"{prefix}.epochs", 40)),
        patience=int(cfg_get(cfg, f"{prefix}.patience", 8)),
        validation_fraction=float(cfg_get(
            cfg, f"{prefix}.validation_fraction", 0.20)),
        random_state=seed,
        num_threads=int(cfg_get(cfg, f"{prefix}.num_threads", 1)),
        verbose=bool(cfg_get(cfg, f"{prefix}.verbose", False)),
    )
