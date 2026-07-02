"""Configuration loading, seeding, and small path helpers.

The whole pipeline is driven by ``config.yaml``. Modules receive the parsed dict
(``cfg``) and read nested values with :func:`cfg_get`, e.g.
``cfg_get(cfg, "eeg.l_freq", 1.0)``.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Project root = parent of this src/ directory.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load ``config.yaml`` (or a custom path) into a plain dict."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(ROOT)
    return cfg


def cfg_get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Fetch a nested config value using a dotted path.

    >>> cfg_get(cfg, "eeg.l_freq", 1.0)
    """
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def resolve_path(cfg: dict, dotted: str) -> Path:
    """Resolve a path from config relative to the project root."""
    raw = cfg_get(cfg, dotted)
    if raw is None:
        raise KeyError(f"config path '{dotted}' is not set")
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def ensure_dirs(cfg: dict) -> None:
    """Create the results/cache directories if they do not exist."""
    for key in ("paths.cache_dir", "paths.results_dir",
                "paths.figures_dir", "paths.metrics_dir"):
        resolve_path(cfg, key).mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy (and torch if present) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # optional
        import torch

        torch.manual_seed(seed)
    except Exception:  # pragma: no cover - torch is optional
        pass
