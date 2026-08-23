"""Selectable classifiers for the motor-imagery pipeline."""

from .ann import MstAnnClassifier
from .cnn import TinyCnnClassifier
from .svm import build_rbf_svm, get_c_gamma, tune_rbf_svm
from .tree import build_depth4_tree

__all__ = [
    "MstAnnClassifier",
    "TinyCnnClassifier",
    "build_depth4_tree",
    "build_rbf_svm",
    "get_c_gamma",
    "tune_rbf_svm",
]
