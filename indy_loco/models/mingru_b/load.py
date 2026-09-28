"""Load a verified, selected EMA fold for normalized-window inference."""

import hashlib
import json
from pathlib import Path

import torch

from .model import ModelB

ROOT = Path(__file__).resolve().parent


def load_fold(session, fold, device="cpu"):
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    records = [
        r for r in manifest["folds"] if r["session"] == session and r["fold"] == fold
    ]
    if len(records) != 1:
        raise ValueError("Unknown session/fold")
    record = records[0]
    path = ROOT / record["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location=device, weights_only=True)
    if (payload["session"], payload["fold"], payload["weight_policy"]) != (
        session,
        fold,
        "ema",
    ):
        raise ValueError("Checkpoint identity mismatch")
    model = ModelB().to(device).eval()
    model.load_state_dict(payload["model_state"], strict=True)
    return model, payload["scalers"]


def predict_velocity(model, normalized_windows, scalers):
    """Return physical-unit velocity; input preprocessing is caller-owned."""
    with torch.inference_mode():
        prediction = model(normalized_windows)[:, -1]
        return prediction * scalers["target_std"].to(prediction) + scalers[
            "target_mean"
        ].to(prediction)
