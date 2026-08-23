"""RBF-SVM construction and training-only tuning."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_rbf_svm(
    c: float = 1.0,
    gamma: float | str = "scale",
    standardize: bool = False,
):
    """Create the classifier.

    Xu et al. use an RBF SVM. They do not state feature z-scoring.
    """
    svc = SVC(kernel="rbf", C=c, gamma=gamma)

    if standardize:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svc", svc),
        ])
    return svc


def tune_rbf_svm(
    x: np.ndarray,
    y: np.ndarray,
    c_grid,
    gamma_grid,
    cv_folds: int = 10,
    random_state: int = 2020,
    standardize: bool = False,
):
    """Training-only tuning of C and gamma.

    Paper-derived fact:
        C and gamma are adjusted to obtain a suitable RBF-SVM.

    Reproduction assumption:
        The exact search grid/procedure is not reported, so this code uses
        stratified cross-validated grid search and never uses official test
        labels for model selection.
    """
    base = build_rbf_svm(standardize=standardize)

    if standardize:
        param_grid = {
            "svc__C": list(c_grid),
            "svc__gamma": list(gamma_grid),
        }
    else:
        param_grid = {
            "C": list(c_grid),
            "gamma": list(gamma_grid),
        }

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    grid = GridSearchCV(
        base,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid.fit(x, y)
    return grid


def get_c_gamma(estimator) -> tuple[float, float | str]:
    if hasattr(estimator, "named_steps"):
        svc = estimator.named_steps["svc"]
        return float(svc.C), svc.gamma
    return float(estimator.C), estimator.gamma
