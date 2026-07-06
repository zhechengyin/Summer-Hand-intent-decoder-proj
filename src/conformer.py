"""Compact EEG Conformer (convolutional transformer) for MI benchmarking.

A small, overfitting-resistant version of the EEG Conformer (Song et al., 2022):

    EEG epoch -> temporal conv -> spatial conv -> pooled time tokens
              -> Transformer encoder (few layers) -> mean-pool
    fNIRS features -> small dense embedding (optional)
    concat -> linear classifier

The training loop (per-channel standardisation, decimation, AdamW, early
stopping on a stratified val split) mirrors ``temporal_cnn`` on purpose, so a
CNN-vs-Transformer comparison isolates the ARCHITECTURE, not the recipe.
Supports ``n_fnirs == 0`` for a pure-EEG architecture benchmark.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import cfg_get


def _build_conformer(n_eeg_ch, n_fnirs, n_classes, d_model, n_heads, n_layers,
                     temporal_kernel, pool, pool_stride, ff_mult, dropout,
                     fnirs_hidden):
    import torch
    import torch.nn as nn

    tk = int(temporal_kernel) + (1 - int(temporal_kernel) % 2)  # force odd

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokenizer = nn.Sequential(
                nn.Conv2d(1, d_model, (1, tk), padding=(0, tk // 2), bias=False),
                nn.BatchNorm2d(d_model),
                nn.Conv2d(d_model, d_model, (n_eeg_ch, 1), bias=False),
                nn.BatchNorm2d(d_model),
                nn.ELU(),
                nn.AvgPool2d((1, int(pool)), stride=(1, int(pool_stride))),
                nn.Dropout(float(dropout)),
            )
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=int(n_heads),
                dim_feedforward=int(d_model * ff_mult), dropout=float(dropout),
                activation="gelu", batch_first=True)
            self.transformer = nn.TransformerEncoder(enc, int(n_layers))
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            if n_fnirs > 0:
                self.fnirs = nn.Sequential(
                    nn.Linear(n_fnirs, fnirs_hidden), nn.GELU(),
                    nn.Dropout(float(dropout)))
                head_in = d_model + fnirs_hidden
            else:
                self.fnirs = None
                head_in = d_model
            self.head = nn.Linear(head_in, n_classes)

        def forward(self, eeg, fnirs):
            z = self.tokenizer(eeg)                 # (B, d, 1, T2)
            z = z.squeeze(2).transpose(1, 2)        # (B, T2, d)
            cls = self.cls.expand(z.size(0), -1, -1)
            z = torch.cat([cls, z], dim=1)
            z = self.transformer(z)
            z = z[:, 0]                             # CLS token summary
            if self.fnirs is not None:
                z = torch.cat([z, self.fnirs(fnirs)], dim=1)
            return self.head(z)

    return Net()


class FusedConformerClassifier:
    """EEG Conformer classifier over EEG epochs (+ optional fNIRS features).

    Mirrors ``FusedTemporalCNNClassifier``'s fit/predict interface so it drops
    into the same CV harness.
    """

    def __init__(self, eeg_decimate=5, d_model=32, n_heads=4, n_layers=2,
                 temporal_kernel=25, pool=8, pool_stride=4, ff_mult=2,
                 dropout=0.4, fnirs_hidden=16, lr=1e-3, weight_decay=1e-2,
                 batch_size=32, epochs=80, patience=12, validation_fraction=0.20,
                 random_state=42, num_threads=1, verbose=False):
        for k, v in dict(
                eeg_decimate=eeg_decimate, d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, temporal_kernel=temporal_kernel, pool=pool,
                pool_stride=pool_stride, ff_mult=ff_mult, dropout=dropout,
                fnirs_hidden=fnirs_hidden, lr=lr, weight_decay=weight_decay,
                batch_size=batch_size, epochs=epochs, patience=patience,
                validation_fraction=validation_fraction,
                random_state=random_state, num_threads=num_threads,
                verbose=verbose).items():
            setattr(self, k, v)

    # -- preprocessing (identical recipe to temporal_cnn) -------------------
    def _decimate(self, X):
        X = np.asarray(X, dtype=np.float32)
        step = max(1, int(self.eeg_decimate))
        return X[..., ::step] if step > 1 else X

    def _fit_transform_eeg(self, X):
        X = self._decimate(X)
        self.eeg_mean_ = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        std = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
        self.eeg_std_ = np.maximum(std, 1e-6)
        return ((X - self.eeg_mean_) / self.eeg_std_).astype(np.float32)

    def _transform_eeg(self, X):
        X = self._decimate(X)
        return ((X - self.eeg_mean_) / self.eeg_std_).astype(np.float32)

    def _split_indices(self, y):
        n = len(y)
        n_classes = len(np.unique(y))
        frac = float(self.validation_fraction)
        _, counts = np.unique(y, return_counts=True)
        if frac <= 0 or n < 2 * n_classes or counts.min() < 2:
            return np.arange(n), np.array([], dtype=int)
        val_size = min(max(n_classes, int(round(n * frac))), n - n_classes)
        from sklearn.model_selection import train_test_split
        return train_test_split(np.arange(n), test_size=val_size, stratify=y,
                                random_state=int(self.random_state))

    def fit(self, eeg_X, fnirs_X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if int(self.num_threads) > 0:
            torch.set_num_threads(int(self.num_threads))
        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state))

        eeg = self._fit_transform_eeg(eeg_X)
        fnirs_X = np.asarray(fnirs_X, dtype=np.float32)
        if fnirs_X.shape[1] > 0:
            self.fnirs_scaler_ = StandardScaler()
            fnirs = self.fnirs_scaler_.fit_transform(fnirs_X).astype(np.float32)
        else:
            self.fnirs_scaler_ = None
            fnirs = fnirs_X
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        ye = np.searchsorted(self.classes_, y).astype(np.int64)

        model = _build_conformer(
            eeg.shape[1], fnirs.shape[1], len(self.classes_),
            self.d_model, self.n_heads, self.n_layers, self.temporal_kernel,
            self.pool, self.pool_stride, self.ff_mult, self.dropout,
            self.fnirs_hidden)
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=float(self.lr),
                                weight_decay=float(self.weight_decay))
        tr, va = self._split_indices(ye)

        def loader(idx, shuffle):
            ds = TensorDataset(torch.from_numpy(eeg[idx]).unsqueeze(1),
                               torch.from_numpy(fnirs[idx]),
                               torch.from_numpy(ye[idx]))
            g = torch.Generator(); g.manual_seed(int(self.random_state))
            return DataLoader(ds, batch_size=min(int(self.batch_size), len(ds)),
                              shuffle=shuffle, generator=g if shuffle else None)

        tl = loader(tr, True)
        vl = loader(va, False) if len(va) else None
        best_state, best_loss, stale = None, float("inf"), 0
        for epoch in range(int(self.epochs)):
            model.train()
            tot, cnt = 0.0, 0
            for xe, xf, yb in tl:
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xe, xf), yb)
                loss.backward()
                opt.step()
                tot += float(loss.item()) * len(yb); cnt += len(yb)
            score = tot / max(1, cnt)
            if vl is not None:
                model.eval(); tot, cnt = 0.0, 0
                with torch.no_grad():
                    for xe, xf, yb in vl:
                        tot += float(loss_fn(model(xe, xf), yb).item()) * len(yb)
                        cnt += len(yb)
                score = tot / max(1, cnt)
            if score < best_loss - 1e-5:
                best_loss, stale = score, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                stale += 1
                if int(self.patience) > 0 and stale >= int(self.patience):
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model.eval()
        return self

    def predict_proba(self, eeg_X, fnirs_X):
        import torch
        eeg = self._transform_eeg(eeg_X)
        fnirs_X = np.asarray(fnirs_X, dtype=np.float32)
        fnirs = (self.fnirs_scaler_.transform(fnirs_X).astype(np.float32)
                 if self.fnirs_scaler_ is not None else fnirs_X)
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(eeg).unsqueeze(1),
                                 torch.from_numpy(fnirs))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, eeg_X, fnirs_X):
        return self.classes_[np.argmax(self.predict_proba(eeg_X, fnirs_X), 1)]


def build_conformer_classifier(cfg: dict, prefix: str = "conformer",
                               **overrides) -> FusedConformerClassifier:
    seed = int(cfg_get(cfg, "seed", 42))
    params = dict(
        eeg_decimate=int(cfg_get(cfg, f"{prefix}.eeg_decimate", 5)),
        d_model=int(cfg_get(cfg, f"{prefix}.d_model", 32)),
        n_heads=int(cfg_get(cfg, f"{prefix}.n_heads", 4)),
        n_layers=int(cfg_get(cfg, f"{prefix}.n_layers", 2)),
        temporal_kernel=int(cfg_get(cfg, f"{prefix}.temporal_kernel", 25)),
        pool=int(cfg_get(cfg, f"{prefix}.pool", 8)),
        pool_stride=int(cfg_get(cfg, f"{prefix}.pool_stride", 4)),
        ff_mult=int(cfg_get(cfg, f"{prefix}.ff_mult", 2)),
        dropout=float(cfg_get(cfg, f"{prefix}.dropout", 0.4)),
        fnirs_hidden=int(cfg_get(cfg, f"{prefix}.fnirs_hidden", 16)),
        lr=float(cfg_get(cfg, f"{prefix}.lr", 1e-3)),
        weight_decay=float(cfg_get(cfg, f"{prefix}.weight_decay", 1e-2)),
        batch_size=int(cfg_get(cfg, f"{prefix}.batch_size", 32)),
        epochs=int(cfg_get(cfg, f"{prefix}.epochs", 80)),
        patience=int(cfg_get(cfg, f"{prefix}.patience", 12)),
        random_state=seed,
        num_threads=int(cfg_get(cfg, f"{prefix}.num_threads", 1)),
    )
    params.update(overrides)
    return FusedConformerClassifier(**params)
