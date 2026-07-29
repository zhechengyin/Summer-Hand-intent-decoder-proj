#!/usr/bin/env python3
"""Integrated 60-second compatibility gate and causal Indy decoder runtime.

The warm-up prefix is used for past-only normalization and for both detector
layers.  Decoder output is released only after the gate permits the session.
This module never reads a velocity label and never updates model weights.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.indy_32ch.decoder_state_detector import (
    TwoLayerCompatibilityGate,
    TwoLayerScore,
    extract_decoder_prefix_trace,
)
from models.indy_32ch.features import multiscale_counts
from models.indy_32ch.input_pipeline import (
    apply_feature_stats,
    fit_feature_stats,
    processed_session_path,
)
from models.indy_32ch.model import build_net


CHECKPOINT_PATH = ROOT / "models" / "indy_32ch" / "64x64checkpoint.pt"
DETECTOR_DIR = ROOT / "results" / "indy" / "phase3c_decoder_state_detector"
LAYER1_PATH = DETECTOR_DIR / "phase3c_active_layer1_reference.npz"
LAYER2_PATH = DETECTOR_DIR / "phase3c_active_layer2_reference.npz"
ALPHAS = (1.0, 0.1)
WINDOW_BINS = 50


@dataclass(frozen=True)
class RuntimeResult:
    """One gate decision and any post-warm-up outputs it permits."""

    gate_score: TwoLayerScore
    output_released: bool
    prediction_velocity: np.ndarray | None
    prediction_bin_end_time_s: np.ndarray | None


def choose_device(requested: str):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


class Indy32Runtime:
    """Frozen model with a mandatory two-layer compatibility gate."""

    def __init__(
        self,
        *,
        net,
        gate: TwoLayerCompatibilityGate,
        channels: np.ndarray,
        feature_std_floor: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        device,
        batch_size: int,
    ) -> None:
        self.net = net
        self.gate = gate
        self.channels = channels
        self.feature_std_floor = feature_std_floor
        self.target_mean = target_mean
        self.target_std = target_std
        self.device = device
        self.batch_size = batch_size
        self.observation_bins = gate.layer1.config.observation_bins
        self.bin_seconds = gate.layer1.config.bin_seconds

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: str | Path = CHECKPOINT_PATH,
        layer1_path: str | Path = LAYER1_PATH,
        layer2_path: str | Path = LAYER2_PATH,
        device: str = "auto",
        batch_size: int = 128,
    ) -> "Indy32Runtime":
        """Load and cross-check the frozen decoder and detector artifacts."""
        import torch

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        selected_device = choose_device(device)
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=selected_device,
            weights_only=False,
        )
        config = checkpoint.get("config", {})
        if checkpoint.get("test_policy") != "locked_not_loaded":
            raise ValueError("Checkpoint does not preserve the locked-test policy")
        if checkpoint.get("observation_seconds") != 60:
            raise ValueError("Checkpoint does not use the frozen 60-second warm-up")
        if config.get("bidir") is not False:
            raise ValueError("Runtime refuses a non-causal bidirectional model")

        channels = np.asarray(checkpoint.get("channels"), dtype=np.int64)
        if channels.shape != (32,) or len(np.unique(channels)) != 32:
            raise ValueError("Checkpoint must contain 32 unique channel indices")
        feature_std_floor = np.asarray(
            checkpoint.get("feature_std_floor"), dtype=np.float32
        ).reshape(-1, 1)
        target_mean = np.asarray(
            checkpoint.get("target_mean"), dtype=np.float32
        )
        target_std = np.asarray(
            checkpoint.get("target_std"), dtype=np.float32
        )
        if feature_std_floor.shape != (64, 1):
            raise ValueError("Checkpoint feature variance floor must have shape (64, 1)")
        if target_mean.shape != (2,) or target_std.shape != (2,):
            raise ValueError("Checkpoint target normalization must contain two axes")
        if np.any(target_std <= 0):
            raise ValueError("Checkpoint target standard deviation must be positive")

        gate = TwoLayerCompatibilityGate.load(layer1_path, layer2_path)
        artifact_channels = gate.layer1.selected_channels
        if artifact_channels is None:
            raise ValueError("Layer-1 artifact has no selected-channel mapping")
        if not np.array_equal(channels, artifact_channels):
            raise ValueError("Detector channel mapping does not match the checkpoint")
        expected_bins = int(
            round(
                checkpoint["observation_seconds"]
                / gate.layer1.config.bin_seconds
            )
        )
        if gate.layer1.config.observation_bins != expected_bins:
            raise ValueError("Detector and checkpoint warm-up lengths do not match")

        net = build_net(config, feature_std_floor.shape[0]).to(selected_device)
        net.load_state_dict(checkpoint["model_state"], strict=True)
        net.eval()
        return cls(
            net=net,
            gate=gate,
            channels=channels,
            feature_std_floor=feature_std_floor,
            target_mean=target_mean,
            target_std=target_std,
            device=selected_device,
            batch_size=batch_size,
        )

    def _validate_selected_counts(self, counts: np.ndarray) -> np.ndarray:
        values = np.asarray(counts, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != 32:
            raise ValueError(
                f"Selected counts must have shape (32, bins), got {values.shape}"
            )
        if values.shape[1] < self.observation_bins:
            raise ValueError(
                f"Need at least {self.observation_bins} bins, got {values.shape[1]}"
            )
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Counts must be finite and non-negative")
        return values

    def select_recording_channels(self, counts: np.ndarray) -> np.ndarray:
        """Apply the frozen channel mapping to a full electrode recording."""
        values = np.asarray(counts)
        if values.ndim != 2 or values.shape[0] <= int(self.channels.max()):
            raise ValueError(
                "Full recording must have shape (all_channels, bins) and include "
                f"channel index {int(self.channels.max())}"
            )
        return self._validate_selected_counts(values[self.channels])

    def assess_selected_counts(self, counts: np.ndarray) -> TwoLayerScore:
        """Make one decision at t=60 s using no future bin or target label."""
        selected = self._validate_selected_counts(counts)
        trace = extract_decoder_prefix_trace(
            self.net,
            selected,
            self.feature_std_floor,
            self.target_mean,
            self.target_std,
            self.gate.layer2.config,
            self.device,
        )
        return self.gate.score(selected, trace)

    def assess_recording_counts(self, counts: np.ndarray) -> TwoLayerScore:
        """Select the frozen channels, then run the integrated warm-up gate."""
        return self.assess_selected_counts(self.select_recording_channels(counts))

    def run_selected_counts(
        self,
        counts: np.ndarray,
        *,
        allow_warning: bool = True,
    ) -> RuntimeResult:
        """Gate first, then release only post-warm-up causal predictions."""
        import torch

        selected = self._validate_selected_counts(counts)
        gate_score = self.assess_selected_counts(selected)
        permitted = gate_score.decision == "pass" or (
            allow_warning and gate_score.decision == "warning"
        )
        usable_bins = (
            (selected.shape[1] - self.observation_bins) // WINDOW_BINS
        ) * WINDOW_BINS
        if not permitted or usable_bins == 0:
            return RuntimeResult(gate_score, False, None, None)

        features = multiscale_counts(selected, ALPHAS)
        mean, local_std = fit_feature_stats(
            features,
            observation_bins=self.observation_bins,
        )
        normalized = apply_feature_stats(
            features,
            (mean, np.maximum(local_std, self.feature_std_floor)),
        )
        starts = range(
            self.observation_bins,
            self.observation_bins + usable_bins,
            WINDOW_BINS,
        )
        windows = np.stack(
            [normalized[:, start : start + WINDOW_BINS] for start in starts]
        ).astype(np.float32)

        prediction_batches = []
        self.net.eval()
        with torch.inference_mode():
            for start in range(0, len(windows), self.batch_size):
                batch = torch.from_numpy(
                    windows[start : start + self.batch_size]
                ).to(self.device)
                prediction_batches.append(self.net(batch).cpu().numpy())
        prediction_normalized = np.concatenate(prediction_batches, axis=0)
        prediction = (
            prediction_normalized * self.target_std + self.target_mean
        ).reshape(-1, 2).astype(np.float32)
        first_bin = self.observation_bins
        bin_end_time = (
            np.arange(first_bin, first_bin + len(prediction), dtype=np.float64)
            + 1
        ) * self.bin_seconds
        return RuntimeResult(gate_score, True, prediction, bin_end_time)

    def run_recording_counts(
        self,
        counts: np.ndarray,
        *,
        allow_warning: bool = True,
    ) -> RuntimeResult:
        """Apply channel selection, gate, and post-warm-up decoding in order."""
        return self.run_selected_counts(
            self.select_recording_channels(counts),
            allow_warning=allow_warning,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        required=True,
        help="Canonical processed Indy session name, for offline runtime checks.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--block-on-warning",
        action="store_true",
        help="Do not release output when the gate returns warning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = Indy32Runtime.load(device=args.device)
    path = processed_session_path(args.session)
    with np.load(path, allow_pickle=False) as artifact:
        counts = artifact["counts"].astype(np.float32)
    result = runtime.run_recording_counts(
        counts,
        allow_warning=not args.block_on_warning,
    )
    print("=== Indy integrated compatibility-gated runtime ===")
    print(f"session: {args.session}")
    print(
        "decision: "
        f"combined={result.gate_score.decision} | "
        f"layer1={result.gate_score.layer1.combined_decision} | "
        f"layer2={result.gate_score.layer2.decision}"
    )
    print(
        "output: "
        + (
            f"released {len(result.prediction_velocity)} post-warm-up bins"
            if result.output_released
            else "blocked"
        )
    )


if __name__ == "__main__":
    main()
