#!/usr/bin/env python3
"""Archived Phase 3c: evaluate a decoder-derived compatibility gate.

This experiment reuses the five already-trained Phase-3b outer-fold
checkpoints.  For each held month it:

1. loads the frozen temporary decoder trained without that month;
2. runs inference on only the first 60 seconds of count data;
3. fits hidden/output references and thresholds on reference months only;
4. scores the held month without loading its velocity labels;
5. loads the completed Phase-3b session table only after all scores exist.

The script never trains or updates a decoder.  January remains structurally
forbidden.  Phase-3b R² is used only to evaluate whether this new label-free
signal separates the two already-known failures from the other sessions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.indy_32ch.decoder_state_detector import (  # noqa: E402
    DecoderStateConfig,
    DecoderStateDetector,
    TwoLayerCompatibilityGate,
    extract_decoder_prefix_trace,
)
from models.indy_32ch.drift_detector import (  # noqa: E402
    DetectorConfig,
    DriftDetector,
    assert_pre_january,
    session_month,
)
from models.indy_32ch.input_pipeline import (  # noqa: E402
    load_session_manifest,
    processed_session_path,
)
from models.indy_32ch.model import build_net  # noqa: E402


MODEL_CONFIG_PATH = ROOT / "configs" / "indy_32ch.yaml"
DETECTOR_CONFIG_PATH = ROOT / "configs" / "indy_32ch_detector.yaml"
ACTIVE_CHECKPOINT_PATH = ROOT / "models" / "indy_32ch" / "checkpoint.pt"
PHASE3B_DIR = ROOT / "results" / "indy" / "phase3b_leave_one_month_out"
PHASE3B_SESSION_CSV = PHASE3B_DIR / "phase3b_leave_one_month_out_sessions.csv"
PHASE3B_CHECKPOINT_DIR = PHASE3B_DIR / "checkpoints"
RESULT_DIR = ROOT / "results" / "indy" / "phase3c_decoder_state_detector"
METRICS_PATH = RESULT_DIR / "phase3c_decoder_state_detector_metrics.json"
SESSION_CSV_PATH = RESULT_DIR / "phase3c_decoder_state_detector_sessions.csv"
SENSITIVITY_CSV_PATH = (
    RESULT_DIR / "phase3c_decoder_state_detector_sensitivity.csv"
)
FIGURE_PATH = RESULT_DIR / "phase3c_decoder_state_detector_figure.png"
ACTIVE_LAYER1_PATH = RESULT_DIR / "phase3c_active_layer1_reference.npz"
ACTIVE_LAYER2_PATH = RESULT_DIR / "phase3c_active_layer2_reference.npz"
ACTIVE_GATE_METADATA_PATH = RESULT_DIR / "phase3c_active_gate_metadata.json"

EXPECTED_MONTHS = ("2016-04", "2016-06", "2016-09", "2016-10", "2016-12")
KNOWN_FAILURES = {"indy_20160630_01", "indy_20161013_03"}
EXPECTED_SESSIONS = 33
EXPECTED_SEED = 43
EXPECTED_EPOCH = 7
SENSITIVITY_COMPONENTS = (3, 5, 8)
SENSITIVITY_SHRINKAGE = (0.05, 0.10, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto selects CUDA when available, otherwise CPU; MPS is excluded.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--folds",
        nargs="+",
        choices=EXPECTED_MONTHS,
        default=list(EXPECTED_MONTHS),
    )
    parser.add_argument("--warning-quantile", type=float, default=0.95)
    parser.add_argument("--severe-quantile", type=float, default=0.99)
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip the 3x3 hidden-dimension/covariance robustness check.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate isolation, checkpoints and trace shapes; do not write results.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def choose_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_development_sessions() -> tuple[list[str], dict[str, list[str]]]:
    manifest = load_session_manifest()
    split = manifest["chronological_split"]
    names = list(split["train"]) + list(split["validation"])
    assert_pre_january(names)
    if len(names) != EXPECTED_SESSIONS:
        raise ValueError(f"Expected {EXPECTED_SESSIONS} sessions, found {len(names)}")
    if set(names) & set(split["test"]):
        raise RuntimeError("Development pool intersects the January test split")
    by_month = {
        month: [name for name in names if session_month(name) == month]
        for month in EXPECTED_MONTHS
    }
    if any(not values for values in by_month.values()):
        raise ValueError(f"Missing expected development month: {by_month}")
    return names, by_month


def load_selected_counts(name: str, channels: np.ndarray) -> np.ndarray:
    """Load only counts; the velocity array is never requested."""
    assert_pre_january([name])
    path = processed_session_path(name)
    if path.parent.name == "test":
        raise RuntimeError(f"Refusing to read a test artifact: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        counts = artifact["counts"][channels].astype(np.float32)
    return counts


def load_fold_checkpoint(held_month: str, device) -> tuple[dict, object]:
    import torch

    path = PHASE3B_CHECKPOINT_DIR / f"held_{held_month}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Phase-3b fold checkpoint: {path}. Run Phase 3b first."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("purpose") != "phase3b_temporary_leave_one_month_out_decoder":
        raise ValueError(f"Unexpected checkpoint purpose in {path}")
    if checkpoint.get("held_month") != held_month:
        raise ValueError(f"Checkpoint held-month mismatch in {path}")
    if checkpoint.get("seed") != EXPECTED_SEED:
        raise ValueError(f"Checkpoint seed is not {EXPECTED_SEED}: {path}")
    if checkpoint.get("checkpoint_epoch") != EXPECTED_EPOCH:
        raise ValueError(f"Checkpoint epoch is not {EXPECTED_EPOCH}: {path}")
    if checkpoint.get("january_policy") != "forbidden_not_loaded":
        raise ValueError(f"Checkpoint January policy is not locked: {path}")

    training_names = list(checkpoint["training_sessions"])
    held_names = list(checkpoint["held_sessions"])
    assert_pre_january(training_names + held_names)
    if set(training_names) & set(held_names):
        raise RuntimeError(f"Training/held overlap in {path}")
    if held_month in {session_month(name) for name in training_names}:
        raise RuntimeError(f"Held month leaked into training in {path}")
    if {session_month(name) for name in held_names} != {held_month}:
        raise RuntimeError(f"Checkpoint contains a wrong held session in {path}")

    channels = np.asarray(checkpoint["channels"], dtype=np.int64)
    net = build_net(checkpoint["config"], len(channels) * 2).to(device)
    net.load_state_dict(checkpoint["model_state"])
    net.eval()
    return checkpoint, net


def run_fold(
    held_month: str,
    all_names: list[str],
    by_month: dict[str, list[str]],
    state_config: DecoderStateConfig,
    device,
    *,
    validate_only: bool,
    run_sensitivity: bool,
) -> tuple[list[dict], dict, list[dict]]:
    checkpoint, net = load_fold_checkpoint(held_month, device)
    training_names = list(checkpoint["training_sessions"])
    held_names = list(checkpoint["held_sessions"])
    if set(training_names + held_names) != set(all_names):
        raise RuntimeError(f"Fold {held_month} does not cover the development pool")
    if set(held_names) != set(by_month[held_month]):
        raise RuntimeError(f"Fold {held_month} held-session manifest mismatch")

    channels = np.asarray(checkpoint["channels"], dtype=np.int64)
    feature_std_floor = np.asarray(checkpoint["feature_std_floor"], dtype=np.float32)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)

    print(
        f"\n=== held month {held_month} | reference={len(training_names)} | "
        f"held={len(held_names)} ===",
        flush=True,
    )
    traces = {}
    for index, name in enumerate(all_names, start=1):
        traces[name] = extract_decoder_prefix_trace(
            net,
            load_selected_counts(name, channels),
            feature_std_floor,
            target_mean,
            target_std,
            state_config,
            device,
        )
        print(
            f"  trace {index:02d}/{len(all_names)} {name}",
            end="\r" if index < len(all_names) else "\n",
            flush=True,
        )

    first = traces[all_names[0]]
    if first.hidden_states.shape[0] != state_config.observation_bins:
        raise AssertionError("Hidden trace did not stop at the observation boundary")
    if first.predicted_velocity.shape != (state_config.observation_bins, 2):
        raise AssertionError("Output trace has an unexpected shape")
    if validate_only:
        return [], {
            "held_month": held_month,
            "trace_shape": list(first.hidden_states.shape),
            "output_shape": list(first.predicted_velocity.shape),
        }, []

    detector = DecoderStateDetector(state_config).fit(
        {name: traces[name] for name in training_names}
    )
    rows = []
    for name in held_names:
        score = detector.score(traces[name])
        row = {
            "session": name,
            "held_month": held_month,
            **asdict(score),
        }
        for metric in DecoderStateDetector.METRICS:
            warning = detector.warning_thresholds[metric]
            severe = detector.severe_thresholds[metric]
            row[f"{metric}_warning_threshold"] = warning
            row[f"{metric}_severe_threshold"] = severe
            row[f"{metric}_to_warning"] = (
                float(row[metric]) / warning
                if warning > 0
                else float(float(row[metric]) > 0)
            )
            row[f"{metric}_to_severe"] = (
                float(row[metric]) / severe
                if severe > 0
                else float(float(row[metric]) > 0)
            )
        rows.append(row)
        print(
            f"  held {name} | gate={score.decision:7s} | "
            f"diagnostic={score.diagnostic_decision:7s} | "
            f"hidden-severe={row['hidden_state_kld_to_severe']:.3f}x | "
            f"output-severe={row['output_state_kld_to_severe']:.3f}x | "
            f"diagnostic-temporal={row['chunk_hidden_kld_max_to_warning']:.3f}x",
            flush=True,
        )

    sensitivity_rows = []
    if run_sensitivity:
        for hidden_components in SENSITIVITY_COMPONENTS:
            for shrinkage in SENSITIVITY_SHRINKAGE:
                variant = replace(
                    state_config,
                    hidden_components=hidden_components,
                    covariance_shrinkage=shrinkage,
                )
                variant_detector = DecoderStateDetector(variant).fit(
                    {name: traces[name] for name in training_names}
                )
                variant_name = (
                    f"hidden_{hidden_components}_shrinkage_{shrinkage:.2f}"
                )
                for name in held_names:
                    score = variant_detector.score(traces[name])
                    sensitivity_rows.append(
                        {
                            "variant": variant_name,
                            "hidden_components": hidden_components,
                            "covariance_shrinkage": shrinkage,
                            "session": name,
                            "held_month": held_month,
                            "decision": score.decision,
                            "diagnostic_decision": score.diagnostic_decision,
                            "hidden_state_kld_to_severe": (
                                score.hidden_state_kld
                                / variant_detector.severe_thresholds[
                                    "hidden_state_kld"
                                ]
                            ),
                            "output_state_kld_to_severe": (
                                score.output_state_kld
                                / variant_detector.severe_thresholds[
                                    "output_state_kld"
                                ]
                            ),
                        }
                    )
    return rows, {
        "held_month": held_month,
        "training_sessions": training_names,
        "held_sessions": held_names,
        "checkpoint": str(
            PHASE3B_CHECKPOINT_DIR / f"held_{held_month}.pt"
        ),
        "detector": detector.metadata(),
        "trace_shape": list(first.hidden_states.shape),
        "output_shape": list(first.predicted_velocity.shape),
    }, sensitivity_rows


def attach_phase3b_outcomes(rows: list[dict]) -> pd.DataFrame:
    """Load labels only after every fold has produced detector scores."""
    if not PHASE3B_SESSION_CSV.exists():
        raise FileNotFoundError(f"Missing Phase-3b session table: {PHASE3B_SESSION_CSV}")
    outcomes = pd.read_csv(PHASE3B_SESSION_CSV)
    required = {
        "session",
        "held_month",
        "decoder_r2_mean",
        "combined_decision",
        "combined_evidence_count",
    }
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"Phase-3b session table is missing columns: {sorted(missing)}")
    scores = pd.DataFrame(rows)
    merged = scores.merge(
        outcomes[
            [
                "session",
                "held_month",
                "decoder_r2_mean",
                "combined_decision",
                "combined_evidence_count",
            ]
        ].rename(
            columns={
                "combined_decision": "layer1_decision",
                "combined_evidence_count": "layer1_evidence_count",
            }
        ),
        on=["session", "held_month"],
        how="left",
        validate="one_to_one",
    )
    if merged["decoder_r2_mean"].isna().any():
        raise RuntimeError("A detector score could not be matched to Phase-3b R²")
    if len(merged) != EXPECTED_SESSIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_SESSIONS} merged sessions, found {len(merged)}"
        )
    merged["known_failure"] = merged["session"].isin(KNOWN_FAILURES)
    merged["layer2_flag"] = merged["decision"] != "pass"
    merged["proposed_combined_decision"] = np.select(
        [
            (merged["layer1_decision"] == "abstain")
            | (merged["decision"] == "abstain"),
            (merged["layer1_decision"] == "warning")
            | (merged["decision"] == "warning"),
        ],
        ["abstain", "warning"],
        default="pass",
    )
    return merged


def promotion_audit(
    frame: pd.DataFrame, sensitivity: pd.DataFrame
) -> dict:
    failures = frame[frame["known_failure"]]
    others = frame[~frame["known_failure"]]
    if set(failures["session"]) != KNOWN_FAILURES:
        raise RuntimeError("Known failure sessions are missing from Phase 3c")

    both_flagged = bool(failures["layer2_flag"].all())
    both_abstained = bool((failures["decision"] == "abstain").all())
    other_warnings = int((others["decision"] == "warning").sum())
    other_abstains = int((others["decision"] == "abstain").sum())
    other_diagnostic_flags = int(
        (others["diagnostic_decision"] != "pass").sum()
    )
    zero_other_impact = other_warnings == 0 and other_abstains == 0
    zero_other_abstain = other_abstains == 0

    # Automatic integration is intentionally stricter than "interesting".
    # The same 33 sessions are development data now that their R² is known.
    sensitivity_failures = sensitivity[sensitivity["session"].isin(KNOWN_FAILURES)]
    sensitivity_others = sensitivity[~sensitivity["session"].isin(KNOWN_FAILURES)]
    sensitivity_variants = int(sensitivity["variant"].nunique())
    sensitivity_both_abstained = bool(
        not sensitivity_failures.empty
        and (sensitivity_failures["decision"] == "abstain").all()
    )
    sensitivity_zero_other_impact = bool(
        not sensitivity_others.empty
        and (sensitivity_others["decision"] == "pass").all()
    )
    sensitivity_passed = (
        sensitivity_variants
        == len(SENSITIVITY_COMPONENTS) * len(SENSITIVITY_SHRINKAGE)
        and sensitivity_both_abstained
        and sensitivity_zero_other_impact
    )
    automatic_integration = (
        both_abstained and zero_other_impact and sensitivity_passed
    )
    research_candidate = (
        both_flagged
        and zero_other_abstain
        and other_warnings / max(len(others), 1) <= 0.10
    )
    return {
        "known_failures": sorted(KNOWN_FAILURES),
        "known_failure_decisions": {
            row["session"]: row["decision"]
            for row in failures.to_dict(orient="records")
        },
        "both_known_failures_flagged": both_flagged,
        "both_known_failures_abstained": both_abstained,
        "other_sessions": len(others),
        "other_warnings": other_warnings,
        "other_abstains": other_abstains,
        "other_diagnostic_flags": other_diagnostic_flags,
        "zero_other_impact": zero_other_impact,
        "zero_other_abstain": zero_other_abstain,
        "research_candidate_gate_passed": research_candidate,
        "sensitivity_variants": sensitivity_variants,
        "sensitivity_both_failures_abstained": sensitivity_both_abstained,
        "sensitivity_zero_other_impact": sensitivity_zero_other_impact,
        "sensitivity_gate_passed": sensitivity_passed,
        "automatic_integration_gate_passed": automatic_integration,
        "automatic_integration_rule": (
            "both known failures abstain and every other session remains pass "
            "in the default and all 3x3 sensitivity variants"
        ),
    }


def correlation_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    fields = [
        f"{metric}_to_warning" for metric in DecoderStateDetector.METRICS
    ] + ["evidence_count", "severe_evidence_count"]
    for field in fields:
        statistic, p_value = spearmanr(frame[field], frame["decoder_r2_mean"])
        rows.append(
            {
                "metric": field,
                "spearman_rho_vs_decoder_r2": (
                    None if not np.isfinite(statistic) else float(statistic)
                ),
                "p_value_descriptive_only": (
                    None if not np.isfinite(p_value) else float(p_value)
                ),
            }
        )
    return rows


def fit_active_two_layer_gate(
    all_names: list[str],
    state_config: DecoderStateConfig,
    device,
) -> dict:
    """Fit runtime references after the strict development gate passes.

    The two label-identified Phase-3b failures are development outcomes and
    cannot be treated as examples of the decoder's compatible operating state.
    They are therefore excluded from the final reference fit, while remaining
    available for an honest artifact-level diagnostic audit.
    """
    import torch

    checkpoint = torch.load(
        ACTIVE_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("seed") != EXPECTED_SEED:
        raise ValueError("Active checkpoint is not the frozen seed-43 model")
    if checkpoint.get("checkpoint_epoch") != EXPECTED_EPOCH:
        raise ValueError("Active checkpoint is not the frozen epoch-7 model")
    if checkpoint.get("test_policy") != "locked_not_loaded":
        raise ValueError("Active checkpoint does not preserve the January lock")
    manifest_names = list(checkpoint["train_sessions"]) + list(
        checkpoint["validation_sessions"]
    )
    assert_pre_january(manifest_names)
    if set(manifest_names) != set(all_names):
        raise RuntimeError("Active checkpoint reference manifest changed")
    reference_names = [
        name for name in manifest_names if name not in KNOWN_FAILURES
    ]
    if len(reference_names) != EXPECTED_SESSIONS - len(KNOWN_FAILURES):
        raise RuntimeError("Active compatible-reference exclusion changed")

    channels = np.asarray(checkpoint["channels"], dtype=np.int64)
    net = build_net(checkpoint["config"], len(channels) * 2).to(device)
    net.load_state_dict(checkpoint["model_state"])
    net.eval()
    feature_std_floor = np.asarray(checkpoint["feature_std_floor"], dtype=np.float32)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)

    all_counts = {
        name: load_selected_counts(name, channels) for name in manifest_names
    }
    all_traces = {
        name: extract_decoder_prefix_trace(
            net,
            all_counts[name],
            feature_std_floor,
            target_mean,
            target_std,
            state_config,
            device,
        )
        for name in manifest_names
    }
    with DETECTOR_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        detector_yaml = yaml.safe_load(handle)
    layer1_config = DetectorConfig(
        observation_bins=int(detector_yaml["data_policy"]["observation_bins"]),
        n_components=int(
            detector_yaml["mindful_inspired_detector"]["dimensions"]
        ),
        warning_quantile=float(detector_yaml["thresholds"]["quantile"]),
    )
    gate = TwoLayerCompatibilityGate(
        DriftDetector(layer1_config),
        DecoderStateDetector(state_config),
    ).fit(
        {name: all_counts[name] for name in reference_names},
        {name: all_traces[name] for name in reference_names},
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    gate.layer1.save(ACTIVE_LAYER1_PATH, selected_channels=channels)
    gate.layer2.save(ACTIVE_LAYER2_PATH)
    artifact_audit = {
        name: {
            "decision": score.decision,
            "layer1_decision": score.layer1.combined_decision,
            "layer2_decision": score.layer2.decision,
        }
        for name in manifest_names
        for score in [gate.score(all_counts[name], all_traces[name])]
    }
    metadata = {
        "status": "integrated_runtime_candidate_not_prospectively_frozen",
        "created_at_utc": utc_now(),
        "january_loaded": False,
        "velocity_labels_loaded": False,
        "weights_updated": False,
        "checkpoint": str(ACTIVE_CHECKPOINT_PATH),
        "reference_sessions": reference_names,
        "excluded_development_failures": sorted(KNOWN_FAILURES),
        "reference_selection_note": (
            "Phase-3b label-identified failures are excluded from compatible "
            "references; this makes the artifact development-selected."
        ),
        "active_checkpoint_caveat": (
            "The active decoder was previously trained on both excluded "
            "sessions, unlike each strict Phase-3b held-month decoder."
        ),
        "artifact_level_diagnostic_audit": artifact_audit,
        "gate": gate.metadata(),
        "artifacts": {
            "layer1": str(ACTIVE_LAYER1_PATH),
            "layer2": str(ACTIVE_LAYER2_PATH),
        },
    }
    write_json_atomic(metadata, ACTIVE_GATE_METADATA_PATH)
    return metadata


def save_figure(frame: pd.DataFrame) -> None:
    cache = Path(tempfile.gettempdir()) / "indy_decoder_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"pass": "#2CA02C", "warning": "#FFB000", "abstain": "#D62728"}
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=180)
    for decision, group in frame.groupby("decision"):
        axes[0].scatter(
            group["hidden_state_kld_to_severe"],
            group["decoder_r2_mean"],
            color=colors[decision],
            label=decision,
            s=45,
            alpha=0.85,
        )
    axes[0].axvline(1.0, color="black", linestyle=":", linewidth=1)
    axes[0].axhline(0.0, color="black", linestyle=":", linewidth=1)
    axes[0].set(
        xlabel="Hidden-state KLD / fold severe threshold",
        ylabel="Strict out-of-month decoder R²",
        title="Hidden-state compatibility",
    )
    axes[0].legend()

    axes[1].scatter(
        frame["output_state_kld_to_severe"],
        frame["decoder_r2_mean"],
        c=[colors[value] for value in frame["decision"]],
        s=45,
        alpha=0.85,
    )
    axes[1].axvline(1.0, color="black", linestyle=":", linewidth=1)
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="Absolute-output KLD / fold severe threshold",
        ylabel="Strict out-of-month decoder R²",
        title="Decoder-output compatibility",
    )

    decision_counts = (
        frame["decision"]
        .value_counts()
        .reindex(["pass", "warning", "abstain"], fill_value=0)
    )
    axes[2].bar(
        decision_counts.index,
        decision_counts.values,
        color=[colors[value] for value in decision_counts.index],
    )
    axes[2].set(
        xlabel="Layer-2 decision",
        ylabel="Sessions",
        title="Held-session impact",
    )
    figure.suptitle("Phase 3c — decoder-derived second-layer detector")
    figure.tight_layout()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)


def write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    import torch

    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if not 0.5 < args.warning_quantile < 1.0:
        raise ValueError("--warning-quantile must be between 0.5 and 1")
    if not args.warning_quantile <= args.severe_quantile < 1.0:
        raise ValueError(
            "--severe-quantile must be at least warning-quantile and below 1"
        )
    if len(set(args.folds)) != len(args.folds):
        raise ValueError("--folds cannot contain duplicates")

    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    all_names, by_month = load_development_sessions()
    state_config = DecoderStateConfig(
        warning_quantile=args.warning_quantile,
        severe_quantile=args.severe_quantile,
    )
    authoritative = tuple(args.folds) == EXPECTED_MONTHS

    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        model_config = yaml.safe_load(handle)
    if model_config["training"]["seed"] != EXPECTED_SEED:
        raise ValueError("Active model config is no longer frozen at seed 43")

    print("=== Phase 3c: decoder-derived second-layer detector ===")
    print(
        f"sessions={len(all_names)} | folds={', '.join(args.folds)} | "
        f"device={device} | warning q={args.warning_quantile:.3f} | "
        f"severe q={args.severe_quantile:.3f}"
    )
    print("weights: frozen Phase-3b checkpoints; optimizer: never created")
    print("held input: first 60 seconds of counts only")
    print("January: FORBIDDEN and not loaded")
    if not authoritative:
        print("WARNING: subset run cannot make the automatic integration decision")

    started_at = utc_now()
    started = time.time()
    rows = []
    folds = []
    sensitivity_rows = []
    for held_month in args.folds:
        fold_rows, fold_metadata, fold_sensitivity = run_fold(
            held_month,
            all_names,
            by_month,
            state_config,
            device,
            validate_only=args.validate_only,
            run_sensitivity=not args.skip_sensitivity,
        )
        rows.extend(fold_rows)
        folds.append(fold_metadata)
        sensitivity_rows.extend(fold_sensitivity)

    if args.validate_only:
        print("\nvalidate-only complete: no result was written")
        print("January: FORBIDDEN and not loaded")
        return
    if not authoritative:
        raise RuntimeError(
            "Result writing and promotion audit require all five pre-January folds"
        )

    # This is the first point where decoder performance labels are loaded.
    frame = attach_phase3b_outcomes(rows)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    if args.skip_sensitivity:
        raise RuntimeError(
            "The promotion audit requires sensitivity; remove --skip-sensitivity"
        )
    audit = promotion_audit(frame, sensitivity_frame)
    correlations = correlation_rows(frame)
    active_gate_metadata = None
    if audit["automatic_integration_gate_passed"]:
        print("\nDevelopment gate passed; fitting active two-layer references...")
        active_gate_metadata = fit_active_two_layer_gate(
            all_names,
            state_config,
            device,
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SESSION_CSV_PATH, index=False)
    sensitivity_frame.to_csv(SENSITIVITY_CSV_PATH, index=False)
    save_figure(frame)
    metrics = {
        "phase": "3c",
        "status": "complete_development_evaluation",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "january_loaded": False,
        "velocity_labels_loaded_by_layer2": False,
        "phase3b_r2_loaded_after_all_layer2_scores": True,
        "decoder_weights_updated": False,
        "decoder": {
            "seed": EXPECTED_SEED,
            "checkpoint_epoch": EXPECTED_EPOCH,
            "outer_fold_checkpoints": str(PHASE3B_CHECKPOINT_DIR),
        },
        "config": asdict(state_config),
        "folds": folds,
        "promotion_audit": audit,
        "correlations": correlations,
        "active_gate": active_gate_metadata,
        "decision_summary": [
            {
                "decision": decision,
                "sessions": int(len(group)),
                "mean_r2": float(group["decoder_r2_mean"].mean()),
                "worst_r2": float(group["decoder_r2_mean"].min()),
            }
            for decision, group in frame.groupby("decision", sort=False)
        ],
        "interpretation_guardrails": [
            "The two failures and all other Phase-3b R² values are now development outcomes.",
            "Passing the development gate does not constitute independent validation.",
            "January did not participate and must not be used to tune these metrics.",
            "A pure intent-mapping change can remain invisible to every label-free signal.",
            "Automatic source integration is allowed only by the recorded strict gate.",
        ],
        "artifacts": {
            "metrics": str(METRICS_PATH),
            "session_csv": str(SESSION_CSV_PATH),
            "sensitivity_csv": str(SENSITIVITY_CSV_PATH),
            "figure": str(FIGURE_PATH),
            "active_layer1_reference": (
                str(ACTIVE_LAYER1_PATH)
                if active_gate_metadata is not None
                else None
            ),
            "active_layer2_reference": (
                str(ACTIVE_LAYER2_PATH)
                if active_gate_metadata is not None
                else None
            ),
            "active_gate_metadata": (
                str(ACTIVE_GATE_METADATA_PATH)
                if active_gate_metadata is not None
                else None
            ),
        },
    }
    write_json_atomic(metrics, METRICS_PATH)

    print("\n=== Phase 3c audit ===")
    for name in sorted(KNOWN_FAILURES):
        row = frame[frame["session"] == name].iloc[0]
        print(
            f"  {name} | R2={row['decoder_r2_mean']:+.4f} | "
            f"layer2={row['decision']}"
        )
    print(
        f"  other sessions | warning={audit['other_warnings']} | "
        f"abstain={audit['other_abstains']} | "
        f"diagnostic-only flags={audit['other_diagnostic_flags']}"
    )
    print(
        f"  sensitivity | variants={audit['sensitivity_variants']} | "
        f"failures stable={audit['sensitivity_both_failures_abstained']} | "
        f"others stable={audit['sensitivity_zero_other_impact']}"
    )
    print(
        "  research candidate: "
        f"{audit['research_candidate_gate_passed']} | "
        "automatic integration: "
        f"{audit['automatic_integration_gate_passed']}"
    )
    print(f"\nmetrics: {METRICS_PATH}")
    print(f"sessions: {SESSION_CSV_PATH}")
    print(f"sensitivity: {SENSITIVITY_CSV_PATH}")
    print(f"figure: {FIGURE_PATH}")
    if active_gate_metadata is not None:
        print(f"active gate: {ACTIVE_GATE_METADATA_PATH}")
    print("January: FORBIDDEN and not loaded")


if __name__ == "__main__":
    main()
