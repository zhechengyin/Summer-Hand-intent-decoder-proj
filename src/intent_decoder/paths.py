"""Canonical repository paths."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSING_DIR = DATA_DIR / "processing"
CONFIG_DIR = REPO_ROOT / "configs"
DATASET_CONFIG_DIR = CONFIG_DIR / "datasets"
RESULTS_DIR = REPO_ROOT / "results"
MODEL_DIR = REPO_ROOT / "models"
