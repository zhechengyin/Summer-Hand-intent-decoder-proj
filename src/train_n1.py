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

from .config import cfg_get
from .fusion import FeatureSet


# ---------------------------------------------------------------------------
# Classifier factory + pipeline
# ---------------------------------------------------------------------------
def build_classifier(name: str, seed: int = 42):
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
                     ("clf", build_classifier(name, seed))])


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
