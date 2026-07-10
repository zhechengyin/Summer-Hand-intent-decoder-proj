"""Stage 5 -- N1, the Neural Evidence Decoder.

N1 maps a feature vector (from time-domain EEG/fNIRS windows) to a probability
vector over the four imagined actions, plus a confidence and uncertainty measure.
It is the "AI inference" block of the AI-Spine architecture.

Baselines first (dataset is small: 7 subjects): Logistic Regression, LDA, SVM,
Random Forest, Gradient Boosting. Each is wrapped in a StandardScaler pipeline so
feature scaling is fit on training data only.

A deep-learning branch (temporal CNN) is *proposed* at the bottom but is off by
default -- see ``build_torch_cnn``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .config import cfg_get
from .fusion import FeatureSet


# ---------------------------------------------------------------------------
# Simple GELU neural network (sklearn-compatible PyTorch estimator)
# ---------------------------------------------------------------------------
class TorchGeluMLPClassifier(ClassifierMixin, BaseEstimator):
    """Small PyTorch MLP with GELU activations and sklearn-style methods.

    It deliberately consumes 2-D feature matrices, so it can drop into the same
    StandardScaler -> classifier Pipeline as the classical baselines.
    """

    _estimator_type = "classifier"

    def __init__(self, hidden_layers=(64, 32), dropout=0.10, lr=1e-3,
                 weight_decay=1e-3, batch_size=32, epochs=80, patience=10,
                 validation_fraction=0.15, random_state=42, verbose=False):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose

    def get_params(self, deep=True):
        return {
            "hidden_layers": self.hidden_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "validation_fraction": self.validation_fraction,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _build_model(self, n_features: int, n_classes: int):
        import torch.nn as nn

        layers = []
        prev = n_features
        for width in tuple(self.hidden_layers):
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.GELU())
            if float(self.dropout) > 0:
                layers.append(nn.Dropout(float(self.dropout)))
            prev = int(width)
        layers.append(nn.Linear(prev, n_classes))
        return nn.Sequential(*layers)

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

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_encoded = np.searchsorted(self.classes_, y).astype(np.int64)
        self.n_features_in_ = X.shape[1]

        torch.manual_seed(int(self.random_state))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._build_model(self.n_features_in_, len(self.classes_)).to(device)
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=float(self.lr),
                                weight_decay=float(self.weight_decay))

        train_idx, val_idx = self._split_indices(y_encoded)

        def loader(indices, shuffle):
            ds = TensorDataset(torch.from_numpy(X[indices]),
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
            running, seen = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                running += float(loss.item()) * len(xb)
                seen += len(xb)
            score = running / max(1, seen)

            if val_loader is not None:
                model.eval()
                total, count = 0.0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        loss = loss_fn(model(xb), yb)
                        total += float(loss.item()) * len(xb)
                        count += len(xb)
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

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# Classifier factory + pipeline
# ---------------------------------------------------------------------------
def build_classifier(name: str, seed: int = 42, cfg: dict | None = None):
    """Instantiate a probability-capable scikit-learn classifier by name."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    name = (name or "lda").lower()
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name in ("logreg", "logistic"):
        return LogisticRegression(max_iter=2000, C=1.0)
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                   random_state=seed)
    if name in ("rf", "randomforest"):
        return RandomForestClassifier(n_estimators=300, random_state=seed,
                                      n_jobs=-1)
    if name in ("gb", "gradientboosting"):
        return GradientBoostingClassifier(random_state=seed)
    if name in ("gelu_nn", "gelu-mlp", "torch_mlp", "nn", "mlp"):
        cfg = cfg or {}
        return TorchGeluMLPClassifier(
            hidden_layers=tuple(cfg_get(cfg, "neural_network.hidden_layers",
                                        [64, 32])),
            dropout=float(cfg_get(cfg, "neural_network.dropout", 0.10)),
            lr=float(cfg_get(cfg, "neural_network.lr", 1e-3)),
            weight_decay=float(cfg_get(cfg, "neural_network.weight_decay",
                                       1e-3)),
            batch_size=int(cfg_get(cfg, "neural_network.batch_size", 32)),
            epochs=int(cfg_get(cfg, "neural_network.epochs", 80)),
            patience=int(cfg_get(cfg, "neural_network.patience", 10)),
            validation_fraction=float(cfg_get(
                cfg, "neural_network.validation_fraction", 0.15)),
            random_state=seed,
        )
    if name == "xgb":  # optional, only if xgboost is installed
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1,
                             subsample=0.9, eval_metric="mlogloss",
                             random_state=seed)
    raise ValueError(f"unknown classifier '{name}'")


def build_pipeline(cfg: dict, name: str | None = None):
    """StandardScaler -> classifier Pipeline (prevents train/test leakage)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    name = name or cfg_get(cfg, "model.classifier", "lda")
    seed = int(cfg_get(cfg, "seed", 42))
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", build_classifier(name, seed, cfg))])


# ---------------------------------------------------------------------------
# N1 decoder wrapper (produces the structured probability output)
# ---------------------------------------------------------------------------
@dataclass
class N1Output:
    intent: str
    confidence: float
    probabilities: dict[str, float]     # {'reach':.1,'grasp':.72,...}
    margin: float                       # top1 - top2
    entropy: float                      # normalised Shannon entropy (0..1)


class N1Decoder:
    """A fitted pipeline + the class ordering, with structured prediction."""

    def __init__(self, pipeline, classes: list[str]):
        self.pipeline = pipeline
        self.classes = list(classes)

    # -- training -----------------------------------------------------------
    @classmethod
    def train(cls, fs: FeatureSet, cfg: dict, name: str | None = None) -> "N1Decoder":
        pipe = build_pipeline(cfg, name)
        pipe.fit(fs.X, fs.y)
        return cls(pipe, fs.classes)

    # -- inference ----------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, n_classes) probabilities aligned to ``self.classes``."""
        X = np.atleast_2d(X)
        proba = self.pipeline.predict_proba(X)
        model_classes = list(self.pipeline.classes_)   # ints 0..k-1
        out = np.zeros((X.shape[0], len(self.classes)), dtype=float)
        for j, cls_id in enumerate(model_classes):
            out[:, int(cls_id)] = proba[:, j]
        row_sums = out.sum(axis=1, keepdims=True)
        return out / np.clip(row_sums, 1e-12, None)

    def probability_vector(self, x_row: np.ndarray) -> dict[str, float]:
        """The N1 example output: {'reach':0.10,'grasp':0.72,...} for one trial."""
        p = self.predict_proba(x_row)[0]
        return {c: float(p[i]) for i, c in enumerate(self.classes)}

    def predict_one(self, x_row: np.ndarray) -> N1Output:
        p = self.predict_proba(x_row)[0]
        order = np.argsort(p)[::-1]
        top, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        n = len(p)
        ent = float(-np.sum(p * np.log(p + 1e-12)) / np.log(n))  # 0..1
        return N1Output(
            intent=self.classes[top],
            confidence=float(p[top]),
            probabilities={c: float(p[i]) for i, c in enumerate(self.classes)},
            margin=float(p[top] - p[second]),
            entropy=ent,
        )

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "classes": self.classes}, path)

    @classmethod
    def load(cls, path: str | Path) -> "N1Decoder":
        import joblib

        d = joblib.load(path)
        return cls(d["pipeline"], d["classes"])


def train_n1(fs: FeatureSet, cfg: dict, name: str | None = None) -> N1Decoder:
    """Convenience: fit an N1Decoder on an entire FeatureSet (for demo/replay)."""
    return N1Decoder.train(fs, cfg, name)


# ===========================================================================
# OPTIONAL deep-learning proposal (NOT used by default)
# ===========================================================================
def build_torch_cnn(n_eeg_ch: int, n_times: int, n_classes: int = 4):
    """Proposed temporal CNN for when more data / augmentation is available.

    Architecture (EEGNet-style, single EEG branch):
        temporal conv -> depthwise spatial conv -> separable conv -> GAP -> softmax
    A parallel fNIRS branch (1-D conv over the slow response) can be concatenated
    before the dense head for a learned fusion. Kept minimal and optional; the
    classical baselines above are the recommended first milestone.
    """
    import torch
    import torch.nn as nn

    class TemporalCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal = nn.Conv2d(1, 16, (1, 33), padding=(0, 16))
            self.bn1 = nn.BatchNorm2d(16)
            self.spatial = nn.Conv2d(16, 32, (n_eeg_ch, 1), groups=16)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Linear(32, n_classes)

        def forward(self, x):                      # x: (B, 1, ch, time)
            x = torch.relu(self.bn1(self.temporal(x)))
            x = torch.relu(self.bn2(self.spatial(x)))
            x = self.pool(x).flatten(1)
            return self.head(x)

    return TemporalCNN()
