"""Reusable neural intent-decoder code.

Experiment scripts belong under ``experiments/``.  Code in this package is the
stable implementation surface shared by training, evaluation, and deployment.
"""

from .paths import DATA_DIR, REPO_ROOT

__all__ = ["DATA_DIR", "REPO_ROOT"]
