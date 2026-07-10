"""Temporal Convolutional Network (TCN) + GRU classifier for EEG epochs.

    EEG epoch (ch x time)
      -> pointwise spatial mix
      -> dilated causal TCN residual blocks (local temporal features)
      -> GRU over the TCN time steps (longer-range dynamics)
      -> last hidden state (+ optional fNIRS branch) -> linear classifier

Same trainer recipe as ``conformer`` / ``temporal_cnn`` (per-channel
standardisation, decimation, AdamW, early stopping) so it drops into the same CV
harness. Supports ``n_fnirs == 0`` for a pure-EEG benchmark.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import cfg_get


def _build_tcn_gru(n_ch, n_fnirs, n_classes, filters, kernel, dilations,
                   gru_hidden, dropout, fnirs_hidden, activation="gelu"):
    import torch
    import torch.nn as nn
    from torch.nn.utils import weight_norm

    ACTS = {"gelu": nn.GELU, "relu": nn.ReLU, "elu": nn.ELU, "silu": nn.SiLU}
    Act = ACTS.get(activation, nn.GELU)

    class Chomp(nn.Module):
        def __init__(self, c):
            super().__init__(); self.c = c

        def forward(self, x):
            return x[:, :, :-self.c].contiguous() if self.c > 0 else x

    class TemporalBlock(nn.Module):
        def __init__(self, cin, cout, k, d):
            super().__init__()
            pad = (k - 1) * d
            self.net = nn.Sequential(
                weight_norm(nn.Conv1d(cin, cout, k, padding=pad, dilation=d)),
                Chomp(pad), Act(), nn.Dropout(dropout),
                weight_norm(nn.Conv1d(cout, cout, k, padding=pad, dilation=d)),
                Chomp(pad), Act(), nn.Dropout(dropout))
            self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None
            self.act = Act()

        def forward(self, x):
            res = x if self.down is None else self.down(x)
            return self.act(self.net(x) + res)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial = nn.Sequential(nn.Conv1d(n_ch, filters, 1),
                                         nn.BatchNorm1d(filters), Act())
            blocks, cin = [], filters
            for d in dilations:
                blocks.append(TemporalBlock(cin, filters, kernel, d))
                cin = filters
            self.tcn = nn.Sequential(*blocks)
            self.gru = nn.GRU(filters, gru_hidden, batch_first=True)
            if n_fnirs > 0:
                self.fnirs = nn.Sequential(nn.Linear(n_fnirs, fnirs_hidden),
                                           Act(), nn.Dropout(dropout))
                head_in = gru_hidden + fnirs_hidden
            else:
                self.fnirs = None
                head_in = gru_hidden
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(head_in, n_classes)

        def forward(self, eeg, fnirs):
            z = self.tcn(self.spatial(eeg))         # (B, F, T)
            z = z.transpose(1, 2)                   # (B, T, F)
            _, h = self.gru(z)                      # h: (1, B, H)
            z = self.drop(h[-1])
            if self.fnirs is not None:
                z = torch.cat([z, self.fnirs(fnirs)], dim=1)
            return self.head(z)

    return Net()


class TCNGRUClassifier:
    def __init__(self, eeg_decimate=5, filters=32, kernel=7,
                 dilations=(1, 2, 4, 8), gru_hidden=32, dropout=0.3,
                 fnirs_hidden=16, activation="gelu", lr=1e-3, weight_decay=1e-3,
                 batch_size=32, epochs=80, patience=15, validation_fraction=0.2,
                 random_state=42, num_threads=1):
        for k, v in dict(
                eeg_decimate=eeg_decimate, filters=filters, kernel=kernel,
                dilations=dilations, gru_hidden=gru_hidden, dropout=dropout,
                fnirs_hidden=fnirs_hidden, activation=activation, lr=lr,
                weight_decay=weight_decay, batch_size=batch_size, epochs=epochs,
                patience=patience, validation_fraction=validation_fraction,
                random_state=random_state, num_threads=num_threads).items():
            setattr(self, k, v)

    def _decimate(self, X):
        X = np.asarray(X, dtype=np.float32)
        s = max(1, int(self.eeg_decimate))
        return X[..., ::s] if s > 1 else X

    def _fit_eeg(self, X):
        X = self._decimate(X)
        self.mean_ = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std_ = np.maximum(X.std(axis=(0, 2), keepdims=True), 1e-6).astype(np.float32)
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def _tf_eeg(self, X):
        X = self._decimate(X)
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit(self, eeg_X, fnirs_X, y):
        import torch
        import torch.nn as nn
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset

        if int(self.num_threads) > 0:
            torch.set_num_threads(int(self.num_threads))
        torch.manual_seed(int(self.random_state)); np.random.seed(int(self.random_state))
        eeg = self._fit_eeg(eeg_X)
        fnirs_X = np.asarray(fnirs_X, dtype=np.float32)
        if fnirs_X.shape[1] > 0:
            self.sc_ = StandardScaler(); fnirs = self.sc_.fit_transform(fnirs_X).astype(np.float32)
        else:
            self.sc_ = None; fnirs = fnirs_X
        y = np.asarray(y); self.classes_ = np.unique(y)
        ye = np.searchsorted(self.classes_, y).astype(np.int64)
        net = _build_tcn_gru(eeg.shape[1], fnirs.shape[1], len(self.classes_),
                             self.filters, self.kernel, self.dilations,
                             self.gru_hidden, self.dropout, self.fnirs_hidden,
                             self.activation)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        ce = nn.CrossEntropyLoss()
        tr, va = train_test_split(np.arange(len(ye)), test_size=self.validation_fraction,
                                  stratify=ye, random_state=int(self.random_state))

        def loader(idx, sh):
            ds = TensorDataset(torch.from_numpy(eeg[idx]),
                               torch.from_numpy(fnirs[idx]),
                               torch.from_numpy(ye[idx]))
            g = torch.Generator(); g.manual_seed(int(self.random_state))
            return DataLoader(ds, batch_size=min(self.batch_size, len(ds)),
                              shuffle=sh, generator=g if sh else None)

        tl, vl = loader(tr, True), loader(va, False)
        best, bl, stale = None, 1e9, 0
        for ep in range(int(self.epochs)):
            net.train()
            for xe, xf, yb in tl:
                opt.zero_grad(set_to_none=True)
                ce(net(xe, xf), yb).backward(); opt.step()
            net.eval(); tot, cnt = 0.0, 0
            with torch.no_grad():
                for xe, xf, yb in vl:
                    tot += float(ce(net(xe, xf), yb)) * len(yb); cnt += len(yb)
            score = tot / max(1, cnt)
            if score < bl - 1e-4:
                bl, stale = score, 0
                best = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= int(self.patience):
                    break
        if best:
            net.load_state_dict(best)
        self.net_ = net.eval()
        return self

    def predict_proba(self, eeg_X, fnirs_X):
        import torch
        eeg = self._tf_eeg(eeg_X)
        fnirs_X = np.asarray(fnirs_X, dtype=np.float32)
        fnirs = (self.sc_.transform(fnirs_X).astype(np.float32)
                 if self.sc_ is not None else fnirs_X)
        with torch.no_grad():
            logits = self.net_(torch.from_numpy(eeg), torch.from_numpy(fnirs))
            return torch.softmax(logits, 1).cpu().numpy()

    def predict(self, eeg_X, fnirs_X):
        return self.classes_[np.argmax(self.predict_proba(eeg_X, fnirs_X), 1)]


def build_tcn_gru_classifier(cfg: dict, prefix: str = "tcn_gru", **ov):
    seed = int(cfg_get(cfg, "seed", 42))
    p = dict(eeg_decimate=int(cfg_get(cfg, f"{prefix}.eeg_decimate", 5)),
             filters=int(cfg_get(cfg, f"{prefix}.filters", 32)),
             kernel=int(cfg_get(cfg, f"{prefix}.kernel", 7)),
             gru_hidden=int(cfg_get(cfg, f"{prefix}.gru_hidden", 32)),
             dropout=float(cfg_get(cfg, f"{prefix}.dropout", 0.3)),
             lr=float(cfg_get(cfg, f"{prefix}.lr", 1e-3)),
             epochs=int(cfg_get(cfg, f"{prefix}.epochs", 80)),
             patience=int(cfg_get(cfg, f"{prefix}.patience", 15)),
             random_state=seed)
    p.update(ov)
    return TCNGRUClassifier(**p)
