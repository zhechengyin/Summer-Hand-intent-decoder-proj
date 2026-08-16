"""Label-free second-layer checks derived from the frozen decoder.

The existing drift detector works on the first 60 seconds of raw spike counts.
This module inspects the same prefix after frozen-model inference:

* the GRU hidden-state distribution;
* the predicted-velocity distribution and its first differences;
* the worst 10-second hidden-state segment.

No velocity label is accepted by this module, and no model weight is updated.
The first 60 seconds are a gated warm-up interval, so using the completed
prefix to make one decision at t=60 s remains causal at the decision boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from indy_loco.models.indy_32ch.drift_detector import (
    DetectorScore,
    DriftDetector,
    assert_pre_january,
    session_month,
)
from indy_loco.models.indy_32ch.features import multiscale_counts
from indy_loco.models.indy_32ch.input_pipeline import (
    apply_feature_stats,
    fit_feature_stats,
)

ALPHAS = (1.0, 0.1)


@dataclass(frozen=True)
class DecoderStateConfig:
    observation_bins: int = 1500
    bin_seconds: float = 0.04
    hidden_components: int = 5
    chunk_bins: int = 250
    covariance_shrinkage: float = 0.10
    covariance_floor: float = 1e-4
    warning_quantile: float = 0.95
    severe_quantile: float = 0.99


@dataclass(frozen=True)
class DecoderPrefixTrace:
    """Frozen-decoder values for one completed warm-up prefix."""

    hidden_states: np.ndarray
    predicted_velocity: np.ndarray


@dataclass(frozen=True)
class GaussianSummary:
    mean: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class DecoderStateScore:
    """Reader-facing second-layer score for one session prefix."""

    matched_hidden_reference: str
    matched_output_reference: str
    hidden_state_kld: float
    output_state_kld: float
    output_delta_kld: float
    chunk_hidden_kld_max: float
    hidden_abnormal: bool
    output_state_abnormal: bool
    output_delta_abnormal: bool
    output_abnormal: bool
    temporal_abnormal: bool
    hidden_severe: bool
    output_state_severe: bool
    output_delta_severe: bool
    output_severe: bool
    temporal_severe: bool
    evidence_count: int
    severe_evidence_count: int
    diagnostic_decision: str
    decision: str


@dataclass(frozen=True)
class TwoLayerScore:
    """Combined raw-count and decoder-state gate result."""

    layer1: DetectorScore
    layer2: DecoderStateScore
    decision: str


def _validate_trace(
    trace: DecoderPrefixTrace, config: DecoderStateConfig
) -> DecoderPrefixTrace:
    hidden = np.asarray(trace.hidden_states, dtype=np.float64)
    output = np.asarray(trace.predicted_velocity, dtype=np.float64)
    if hidden.ndim != 2:
        raise ValueError(f"Hidden states must be 2D, got {hidden.shape}")
    if output.ndim != 2 or output.shape[1] != 2:
        raise ValueError(
            f"Predicted velocity must have shape (bins, 2), got {output.shape}"
        )
    if hidden.shape[0] < config.observation_bins:
        raise ValueError(
            f"Need {config.observation_bins} hidden-state bins, got {hidden.shape[0]}"
        )
    if output.shape[0] < config.observation_bins:
        raise ValueError(
            f"Need {config.observation_bins} output bins, got {output.shape[0]}"
        )
    hidden = hidden[: config.observation_bins]
    output = output[: config.observation_bins]
    if not np.isfinite(hidden).all() or not np.isfinite(output).all():
        raise ValueError("Decoder trace must contain only finite values")
    return DecoderPrefixTrace(hidden, output)


def extract_decoder_prefix_trace(
    net,
    selected_counts: np.ndarray,
    feature_std_floor: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    config: DecoderStateConfig,
    device,
) -> DecoderPrefixTrace:
    """Run frozen inference on exactly the first 60 seconds.

    ``selected_counts`` must already contain the frozen 32-channel mapping.
    The normalization is the same completed-prefix normalization used by the
    decoder after warm-up.  No bin after ``observation_bins`` is inspected.
    """
    import torch

    counts = np.asarray(selected_counts, dtype=np.float32)
    if counts.ndim != 2 or counts.shape[0] != 32:
        raise ValueError(
            f"Expected selected counts with shape (32, bins), got {counts.shape}"
        )
    if counts.shape[1] < config.observation_bins:
        raise ValueError(
            f"Need {config.observation_bins} count bins, got {counts.shape[1]}"
        )
    prefix = counts[:, : config.observation_bins]
    features = multiscale_counts(prefix, ALPHAS)
    mean, local_std = fit_feature_stats(
        features, observation_bins=config.observation_bins
    )
    floor = np.asarray(feature_std_floor, dtype=np.float32).reshape(-1, 1)
    if floor.shape != local_std.shape:
        raise ValueError(
            f"Feature floor shape {floor.shape} does not match {local_std.shape}"
        )
    normalized = apply_feature_stats(
        features, (mean, np.maximum(local_std, floor))
    ).astype(np.float32)

    net.eval()
    with torch.no_grad():
        values = torch.from_numpy(normalized[None]).to(device)
        prediction_norm, hidden = net.forward_with_states(values)
    prediction = prediction_norm[0].cpu().numpy() * np.asarray(
        target_std, dtype=np.float32
    ) + np.asarray(target_mean, dtype=np.float32)
    return _validate_trace(
        DecoderPrefixTrace(
            hidden_states=hidden[0].cpu().numpy(),
            predicted_velocity=prediction,
        ),
        config,
    )


def _regularized_gaussian(
    samples: np.ndarray, config: DecoderStateConfig
) -> GaussianSummary:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(
            f"Gaussian samples must be a non-trivial 2D array: {values.shape}"
        )
    covariance = np.cov(values, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.asarray([[float(covariance)]])
    diagonal = np.diag(np.diag(covariance))
    covariance = (
        (1.0 - config.covariance_shrinkage) * covariance
        + config.covariance_shrinkage * diagonal
        + config.covariance_floor * np.eye(covariance.shape[0])
    )
    return GaussianSummary(values.mean(axis=0), covariance)


def _symmetric_kld(left: GaussianSummary, right: GaussianSummary) -> float:
    """Symmetric Gaussian KLD for a direction-independent compatibility score."""

    def directional(reference: GaussianSummary, current: GaussianSummary) -> float:
        dimension = reference.mean.size
        difference = current.mean - reference.mean
        sign_reference, logdet_reference = np.linalg.slogdet(reference.covariance)
        sign_current, logdet_current = np.linalg.slogdet(current.covariance)
        if sign_reference <= 0 or sign_current <= 0:
            raise ValueError("Regularized covariance must be positive definite")
        solved_covariance = np.linalg.solve(current.covariance, reference.covariance)
        solved_difference = np.linalg.solve(current.covariance, difference)
        return float(
            max(
                0.5
                * (
                    np.trace(solved_covariance)
                    + difference @ solved_difference
                    - dimension
                    + logdet_current
                    - logdet_reference
                ),
                0.0,
            )
        )

    return 0.5 * (directional(left, right) + directional(right, left))


def _higher_quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calibrate a threshold from no values")
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility
        return float(np.quantile(values, quantile, interpolation="higher"))


class DecoderStateDetector:
    """Multi-reference detector operating on frozen-decoder traces."""

    METRICS = (
        "hidden_state_kld",
        "output_state_kld",
        "output_delta_kld",
        "chunk_hidden_kld_max",
    )

    def __init__(self, config: DecoderStateConfig | None = None) -> None:
        self.config = config if config is not None else DecoderStateConfig()
        self.reference_session_names: tuple[str, ...] = ()
        self.reference_months: tuple[str, ...] = ()
        self.hidden_mean: np.ndarray | None = None
        self.hidden_std: np.ndarray | None = None
        self.hidden_components: np.ndarray | None = None
        self.hidden_month_gaussians: dict[str, GaussianSummary] = {}
        self.output_month_gaussians: dict[str, GaussianSummary] = {}
        self.output_delta_month_gaussians: dict[str, GaussianSummary] = {}
        self.warning_thresholds: dict[str, float] = {}
        self.severe_thresholds: dict[str, float] = {}

    def _traces(
        self, traces: Mapping[str, DecoderPrefixTrace]
    ) -> dict[str, DecoderPrefixTrace]:
        names = tuple(sorted(traces))
        assert_pre_january(names)
        if len(names) < 3:
            raise ValueError("At least three decoder traces are required")
        return {name: _validate_trace(traces[name], self.config) for name in names}

    def _fit_hidden_projection(
        self, traces: Mapping[str, DecoderPrefixTrace]
    ) -> dict[str, np.ndarray]:
        # Every session contributes exactly 1500 bins.
        raw = np.concatenate([trace.hidden_states for trace in traces.values()], axis=0)
        mean = raw.mean(axis=0)
        std = np.maximum(raw.std(axis=0), 1e-6)
        normalized = (raw - mean) / std
        covariance = np.cov(normalized, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1][: self.config.hidden_components]
        components = eigenvectors[:, order].T
        self.hidden_mean = mean
        self.hidden_std = std
        self.hidden_components = components
        return {
            name: ((trace.hidden_states - mean) / std) @ components.T
            for name, trace in traces.items()
        }

    def _fit_reference_state(self, traces: Mapping[str, DecoderPrefixTrace]) -> None:
        latent = self._fit_hidden_projection(traces)
        months = sorted({session_month(name) for name in traces})
        self.reference_months = tuple(months)
        self.hidden_month_gaussians = {
            month: _regularized_gaussian(
                np.concatenate(
                    [
                        values
                        for name, values in latent.items()
                        if session_month(name) == month
                    ],
                    axis=0,
                ),
                self.config,
            )
            for month in months
        }
        self.output_month_gaussians = {
            month: _regularized_gaussian(
                np.concatenate(
                    [
                        trace.predicted_velocity
                        for name, trace in traces.items()
                        if session_month(name) == month
                    ],
                    axis=0,
                ),
                self.config,
            )
            for month in months
        }
        self.output_delta_month_gaussians = {
            month: _regularized_gaussian(
                np.concatenate(
                    [
                        np.diff(trace.predicted_velocity, axis=0)
                        for name, trace in traces.items()
                        if session_month(name) == month
                    ],
                    axis=0,
                ),
                self.config,
            )
            for month in months
        }

    def _project_hidden(self, hidden: np.ndarray) -> np.ndarray:
        if (
            self.hidden_mean is None
            or self.hidden_std is None
            or self.hidden_components is None
        ):
            raise RuntimeError("Decoder-state detector has not been fitted")
        return (
            (hidden - self.hidden_mean) / self.hidden_std
        ) @ self.hidden_components.T

    def _raw_metrics(self, trace: DecoderPrefixTrace) -> dict[str, float | str]:
        if not self.hidden_month_gaussians:
            raise RuntimeError("Decoder-state detector has not been fitted")
        values = _validate_trace(trace, self.config)
        hidden = self._project_hidden(values.hidden_states)
        hidden_gaussian = _regularized_gaussian(hidden, self.config)
        hidden_distances = {
            month: _symmetric_kld(reference, hidden_gaussian)
            for month, reference in self.hidden_month_gaussians.items()
        }
        matched_hidden = min(hidden_distances, key=hidden_distances.get)

        output_gaussian = _regularized_gaussian(values.predicted_velocity, self.config)
        output_delta_gaussian = _regularized_gaussian(
            np.diff(values.predicted_velocity, axis=0), self.config
        )
        output_pairs = {
            month: (
                _symmetric_kld(reference, output_gaussian),
                _symmetric_kld(
                    self.output_delta_month_gaussians[month],
                    output_delta_gaussian,
                ),
            )
            for month, reference in self.output_month_gaussians.items()
        }
        # Choose one coherent historical state by the worst of the two output
        # distances; the raw distances remain separate for calibration.
        matched_output = min(
            output_pairs,
            key=lambda month: max(output_pairs[month]),
        )

        chunk_distances = []
        for start in range(0, self.config.observation_bins, self.config.chunk_bins):
            stop = min(start + self.config.chunk_bins, self.config.observation_bins)
            if stop - start < 2:
                continue
            chunk_gaussian = _regularized_gaussian(hidden[start:stop], self.config)
            chunk_distances.append(
                min(
                    _symmetric_kld(reference, chunk_gaussian)
                    for reference in self.hidden_month_gaussians.values()
                )
            )
        if not chunk_distances:
            raise ValueError("No complete hidden-state chunks were available")

        return {
            "matched_hidden_reference": matched_hidden,
            "matched_output_reference": matched_output,
            "hidden_state_kld": hidden_distances[matched_hidden],
            "output_state_kld": output_pairs[matched_output][0],
            "output_delta_kld": output_pairs[matched_output][1],
            "chunk_hidden_kld_max": max(chunk_distances),
        }

    def _nested_month_calibration(
        self, traces: Mapping[str, DecoderPrefixTrace]
    ) -> list[dict[str, float | str]]:
        months = sorted({session_month(name) for name in traces})
        if len(months) < 3:
            raise ValueError(
                "Nested month calibration requires at least three reference months"
            )
        rows: list[dict[str, float | str]] = []
        for held_month in months:
            inner_reference = {
                name: trace
                for name, trace in traces.items()
                if session_month(name) != held_month
            }
            inner_held = {
                name: trace
                for name, trace in traces.items()
                if session_month(name) == held_month
            }
            inner = DecoderStateDetector(self.config)
            inner._fit_reference_state(inner_reference)
            rows.extend(inner._raw_metrics(trace) for trace in inner_held.values())
        return rows

    def fit(self, traces: Mapping[str, DecoderPrefixTrace]) -> DecoderStateDetector:
        validated = self._traces(traces)
        self.reference_session_names = tuple(sorted(validated))
        self._fit_reference_state(validated)
        calibration = self._nested_month_calibration(validated)
        self.warning_thresholds = {
            name: _higher_quantile(
                [float(row[name]) for row in calibration],
                self.config.warning_quantile,
            )
            for name in self.METRICS
        }
        self.severe_thresholds = {
            name: _higher_quantile(
                [float(row[name]) for row in calibration],
                self.config.severe_quantile,
            )
            for name in self.METRICS
        }
        return self

    @staticmethod
    def _decision(evidence_count: int) -> str:
        if evidence_count >= 2:
            return "abstain"
        if evidence_count == 1:
            return "warning"
        return "pass"

    def score(self, trace: DecoderPrefixTrace) -> DecoderStateScore:
        if not self.warning_thresholds or not self.severe_thresholds:
            raise RuntimeError("Decoder-state detector has not been fitted")
        metrics = self._raw_metrics(trace)

        hidden_abnormal = (
            float(metrics["hidden_state_kld"])
            > self.warning_thresholds["hidden_state_kld"]
        )
        output_state_abnormal = (
            float(metrics["output_state_kld"])
            > self.warning_thresholds["output_state_kld"]
        )
        output_delta_abnormal = (
            float(metrics["output_delta_kld"])
            > self.warning_thresholds["output_delta_kld"]
        )
        output_abnormal = output_state_abnormal or output_delta_abnormal
        temporal_abnormal = (
            float(metrics["chunk_hidden_kld_max"])
            > self.warning_thresholds["chunk_hidden_kld_max"]
        )
        hidden_severe = (
            float(metrics["hidden_state_kld"])
            > self.severe_thresholds["hidden_state_kld"]
        )
        output_state_severe = (
            float(metrics["output_state_kld"])
            > self.severe_thresholds["output_state_kld"]
        )
        output_delta_severe = (
            float(metrics["output_delta_kld"])
            > self.severe_thresholds["output_delta_kld"]
        )
        output_severe = output_state_severe or output_delta_severe
        temporal_severe = (
            float(metrics["chunk_hidden_kld_max"])
            > self.severe_thresholds["chunk_hidden_kld_max"]
        )
        evidence_count = (
            int(hidden_abnormal) + int(output_abnormal) + int(temporal_abnormal)
        )
        severe_count = int(hidden_severe) + int(output_severe) + int(temporal_severe)
        diagnostic_decision = self._decision(evidence_count)
        # The deployed second layer is a conservative veto, not another broad
        # warning generator.  Output-delta and chunk scores remain diagnostics:
        # both produced behavior-dependent false alarms in development.  A
        # session is vetoed only when the decoder's internal representation and
        # its absolute output distribution independently exceed the severe
        # reference-only thresholds.
        gate_decision = "abstain" if hidden_severe and output_state_severe else "pass"
        return DecoderStateScore(
            matched_hidden_reference=str(metrics["matched_hidden_reference"]),
            matched_output_reference=str(metrics["matched_output_reference"]),
            hidden_state_kld=float(metrics["hidden_state_kld"]),
            output_state_kld=float(metrics["output_state_kld"]),
            output_delta_kld=float(metrics["output_delta_kld"]),
            chunk_hidden_kld_max=float(metrics["chunk_hidden_kld_max"]),
            hidden_abnormal=hidden_abnormal,
            output_state_abnormal=output_state_abnormal,
            output_delta_abnormal=output_delta_abnormal,
            output_abnormal=output_abnormal,
            temporal_abnormal=temporal_abnormal,
            hidden_severe=hidden_severe,
            output_state_severe=output_state_severe,
            output_delta_severe=output_delta_severe,
            output_severe=output_severe,
            temporal_severe=temporal_severe,
            evidence_count=evidence_count,
            severe_evidence_count=severe_count,
            diagnostic_decision=diagnostic_decision,
            decision=gate_decision,
        )

    def metadata(self) -> dict:
        return {
            "method": "decoder_hidden_output_multi_reference_gate",
            "config": asdict(self.config),
            "reference_sessions": list(self.reference_session_names),
            "reference_months": list(self.reference_months),
            "warning_thresholds": self.warning_thresholds,
            "severe_thresholds": self.severe_thresholds,
            "threshold_calibration": (
                "nested_leave_one_complete_month_out_on_frozen_decoder_traces"
            ),
            "decision_policy": {
                "diagnostic_warning": "one decoder-state evidence family",
                "diagnostic_abstain": ("at least two decoder-state evidence families"),
                "gate_abstain": (
                    "hidden-state severe and absolute-output-state severe"
                ),
                "output_delta_and_temporal": "diagnostic_only",
                "velocity_labels_used": False,
                "weights_updated": False,
            },
        }

    def save(self, path) -> None:
        """Save fitted second-layer references without pickle."""
        if (
            self.hidden_mean is None
            or self.hidden_std is None
            or self.hidden_components is None
            or not self.hidden_month_gaussians
        ):
            raise RuntimeError("Decoder-state detector has not been fitted")
        months = list(self.reference_months)
        arrays = {
            "metadata_json": np.asarray(
                __import__("json").dumps(self.metadata(), sort_keys=True)
            ),
            "month_names": np.asarray(months),
            "hidden_normalization_mean": self.hidden_mean.astype(np.float32),
            "hidden_normalization_std": self.hidden_std.astype(np.float32),
            "hidden_pca_components": self.hidden_components.astype(np.float32),
            "hidden_month_means": np.stack(
                [self.hidden_month_gaussians[month].mean for month in months]
            ).astype(np.float32),
            "hidden_month_covariances": np.stack(
                [self.hidden_month_gaussians[month].covariance for month in months]
            ).astype(np.float32),
            "output_month_means": np.stack(
                [self.output_month_gaussians[month].mean for month in months]
            ).astype(np.float32),
            "output_month_covariances": np.stack(
                [self.output_month_gaussians[month].covariance for month in months]
            ).astype(np.float32),
            "output_delta_month_means": np.stack(
                [self.output_delta_month_gaussians[month].mean for month in months]
            ).astype(np.float32),
            "output_delta_month_covariances": np.stack(
                [
                    self.output_delta_month_gaussians[month].covariance
                    for month in months
                ]
            ).astype(np.float32),
        }
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> DecoderStateDetector:
        """Restore a fitted second layer from a non-pickle artifact."""
        artifact = Path(path)
        required = {
            "metadata_json",
            "month_names",
            "hidden_normalization_mean",
            "hidden_normalization_std",
            "hidden_pca_components",
            "hidden_month_means",
            "hidden_month_covariances",
            "output_month_means",
            "output_month_covariances",
            "output_delta_month_means",
            "output_delta_month_covariances",
        }
        with np.load(artifact, allow_pickle=False) as arrays:
            missing = required - set(arrays.files)
            if missing:
                raise ValueError(
                    f"Layer-2 artifact is missing arrays: {sorted(missing)}"
                )
            metadata = json.loads(str(np.asarray(arrays["metadata_json"]).item()))
            detector = cls(DecoderStateConfig(**metadata["config"]))
            detector.reference_session_names = tuple(
                str(value) for value in metadata["reference_sessions"]
            )
            detector.reference_months = tuple(
                str(value) for value in metadata["reference_months"]
            )
            detector.warning_thresholds = {
                str(name): float(value)
                for name, value in metadata["warning_thresholds"].items()
            }
            detector.severe_thresholds = {
                str(name): float(value)
                for name, value in metadata["severe_thresholds"].items()
            }
            months = tuple(str(value) for value in arrays["month_names"].tolist())
            if months != detector.reference_months:
                raise ValueError("Layer-2 month arrays do not match artifact metadata")

            detector.hidden_mean = np.asarray(
                arrays["hidden_normalization_mean"], dtype=np.float64
            )
            detector.hidden_std = np.asarray(
                arrays["hidden_normalization_std"], dtype=np.float64
            )
            detector.hidden_components = np.asarray(
                arrays["hidden_pca_components"], dtype=np.float64
            )

            def summaries(mean_key: str, covariance_key: str):
                means = np.asarray(arrays[mean_key], dtype=np.float64)
                covariances = np.asarray(arrays[covariance_key], dtype=np.float64)
                if len(means) != len(months) or len(covariances) != len(months):
                    raise ValueError(
                        f"Layer-2 {mean_key}/{covariance_key} month count mismatch"
                    )
                return {
                    month: GaussianSummary(means[index], covariances[index])
                    for index, month in enumerate(months)
                }

            detector.hidden_month_gaussians = summaries(
                "hidden_month_means", "hidden_month_covariances"
            )
            detector.output_month_gaussians = summaries(
                "output_month_means", "output_month_covariances"
            )
            detector.output_delta_month_gaussians = summaries(
                "output_delta_month_means",
                "output_delta_month_covariances",
            )

        if detector.hidden_components.shape[0] != detector.config.hidden_components:
            raise ValueError(
                "Layer-2 PCA dimension does not match artifact configuration"
            )
        if detector.hidden_mean.shape != detector.hidden_std.shape:
            raise ValueError("Layer-2 hidden normalization arrays do not align")
        if detector.hidden_components.shape[1] != detector.hidden_mean.size:
            raise ValueError(
                "Layer-2 PCA width does not match hidden-state normalization"
            )
        return detector


class TwoLayerCompatibilityGate:
    """Active compatibility gate joining raw and decoder-derived evidence."""

    def __init__(
        self,
        layer1: DriftDetector,
        layer2: DecoderStateDetector,
    ) -> None:
        self.layer1 = layer1
        self.layer2 = layer2

    def fit(
        self,
        count_sessions: Mapping[str, np.ndarray],
        trace_sessions: Mapping[str, DecoderPrefixTrace],
    ) -> TwoLayerCompatibilityGate:
        if set(count_sessions) != set(trace_sessions):
            raise ValueError("Layer-1 counts and layer-2 traces must share sessions")
        self.layer1.fit(count_sessions)
        self.layer2.fit(trace_sessions)
        return self

    def score(
        self,
        counts: np.ndarray,
        trace: DecoderPrefixTrace,
    ) -> TwoLayerScore:
        layer1_score = self.layer1.score(counts)
        layer2_score = self.layer2.score(trace)
        if (
            layer1_score.combined_decision == "abstain"
            or layer2_score.decision == "abstain"
        ):
            decision = "abstain"
        elif layer1_score.combined_decision == "warning":
            decision = "warning"
        else:
            decision = "pass"
        return TwoLayerScore(layer1_score, layer2_score, decision)

    def metadata(self) -> dict:
        return {
            "method": "two_layer_count_plus_frozen_decoder_state_gate",
            "layer1": self.layer1.metadata(),
            "layer2": self.layer2.metadata(),
            "decision_policy": {
                "abstain": "either layer independently abstains",
                "warning": "layer1 warns while neither layer abstains",
                "layer2_warning_policy": "diagnostic_only",
            },
        }

    @classmethod
    def load(
        cls,
        layer1_path: str | Path,
        layer2_path: str | Path,
    ) -> TwoLayerCompatibilityGate:
        """Load and cross-check the two fitted deployment layers."""
        layer1 = DriftDetector.load(layer1_path)
        layer2 = DecoderStateDetector.load(layer2_path)
        if layer1.config.observation_bins != layer2.config.observation_bins:
            raise ValueError("Detector layers use different warm-up lengths")
        if layer1.reference_session_names != layer2.reference_session_names:
            raise ValueError("Detector layers use different reference sessions")
        if layer1.reference_months != layer2.reference_months:
            raise ValueError("Detector layers use different reference months")
        return cls(layer1, layer2)
