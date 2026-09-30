"""Reuse and verify active frozen preprocessing; never import archived code."""

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from indy_loco.experiment.phase16_parameter_scaling import protocol, session_data

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INDY = REPO / "indy_loco"
FROZEN = HERE.parent / "phase16_parameter_scaling"
REFERENCE = (
    HERE.parent / "phase13_deployment_validation/results/rolling_retrain/final_30fold"
)


def locked_ast_dump(node):
    # Python 3.13+ defaults to omitting empty lists; the original lock includes them.
    options = (
        {"show_empty": True}
        if "show_empty" in inspect.signature(ast.dump).parameters
        else {}
    )
    return ast.dump(node, include_attributes=False, **options)


def verify_protocol_lock():
    lock = json.loads((FROZEN / "protocol_lock.json").read_text(encoding="utf-8"))
    for filename, section in (
        ("session_data.py", "session_data_nodes"),
        ("protocol.py", "protocol_nodes"),
    ):
        tree = ast.parse((FROZEN / filename).read_text(encoding="utf-8"))
        nodes = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                nodes[node.name] = node
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                nodes[node.target.id] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        nodes[target.id] = node
        for key, expected in lock[section].items():
            actual = hashlib.sha256(locked_ast_dump(nodes[key]).encode()).hexdigest()
            if actual != expected:
                raise ValueError(f"Frozen preprocessing changed: {filename}:{key}")
        if filename == "protocol.py":
            statements = nodes["fit_model"].body[3:-1]
            digest = hashlib.sha256(
                "\n".join(locked_ast_dump(n) for n in statements).encode()
            ).hexdigest()
            if digest != lock["training_core_sha256"]:
                raise ValueError("Frozen Phase-13 training reference changed")
    return lock


def reference_path(session, fold):
    directory = INDY / "models/midsize" / session
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    entry = next(row for row in manifest["model"]["checkpoints"] if row["fold"] == fold)
    path = directory / entry["file"]
    if protocol.sha256_file(path) != entry["sha256"]:
        raise ValueError(f"Midsize checkpoint SHA mismatch: {path}")
    return path


def load_reference(session, fold):
    path = reference_path(session, fold)
    result = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "session": session,
        "fold": fold,
        "seed": 43,
        "parameter_count": 86978,
        "initialization": "phase7",
        "train_scope": "all",
        "selection_policy": "minimum_validation_loss_test_opened_once",
        "test_evaluated_during_training": False,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Not a canonical final Midsize checkpoint: {path}")
    policy = result["deployment_policy"]
    if (policy["calibration_bins"], policy["window_bins"], policy["ewma_alpha"]) != (
        10500,
        50,
        0.1,
    ):
        raise ValueError("Reference deployment preprocessing changed")
    return result


def preprocessing_evidence(fold_data, reference):
    fields = {
        "channels": "selected_channel_indices",
        "calibration_mean": "feature_mean",
        "calibration_effective_std": "feature_std",
        "calibration_local_std": "calibration_local_std",
        "feature_std_floor": "feature_std_floor",
        "target_mean": "target_mean",
        "target_std": "target_std",
    }
    for field, key in fields.items():
        if not np.array_equal(
            np.asarray(getattr(fold_data, field)).reshape(-1),
            np.asarray(reference[key]).reshape(-1),
        ):
            raise ValueError(f"Preprocessing differs from Midsize: {field}")
    for split in ("train", "validation", "test"):
        if (
            len(getattr(fold_data, f"{split}_bins"))
            != reference["bin_counts_after_calibration"][split]
        ):
            raise ValueError(f"Split bin count changed: {split}")
        if (
            len(getattr(fold_data, f"{split}_reaches"))
            != reference["reach_counts"][split]
        ):
            raise ValueError(f"Split reach count changed: {split}")
    # Velocity identity is needed for safe resume, in addition to target scaling.
    fields = (
        *fields,
        "velocity",
        "normalized_features",
        "train_reaches",
        "validation_reaches",
        "test_reaches",
        "train_bins",
        "validation_bins",
        "test_bins",
    )
    evidence = {}
    for name in fields:
        array = np.ascontiguousarray(getattr(fold_data, name))
        evidence[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    return evidence


def prepare_verified_fold(data, fold, reference):
    """Recompute original folds, audit FP32 roundoff, then reuse frozen scalers.

    ARM/x86 NumPy reductions can differ by a few FP32 ULPs. Integer channels,
    reach/bin counts remain exact; material numerical drift is a hard error.
    No preprocessing is refit using validation/test labels.
    """
    prepared = protocol.prepare_fold(data, fold - 1)
    fields = {
        "calibration_mean": "feature_mean",
        "calibration_effective_std": "feature_std",
        "calibration_local_std": "calibration_local_std",
        "feature_std_floor": "feature_std_floor",
        "target_mean": "target_mean",
        "target_std": "target_std",
    }
    roundoff = {}
    for field, key in fields.items():
        current = np.asarray(getattr(prepared, field)).reshape(-1)
        frozen = np.asarray(reference[key], dtype=np.float32).reshape(-1)
        if current.shape != frozen.shape or not np.allclose(
            current, frozen, rtol=5e-7, atol=5e-8
        ):
            raise ValueError(f"Preprocessing differs beyond FP32 roundoff: {field}")
        roundoff[field] = {
            "max_absolute_difference": float(np.max(np.abs(current - frozen))),
            "exact_before_pinning": bool(np.array_equal(current, frozen)),
        }
        setattr(prepared, field, frozen.copy())
    features = protocol.continuous_features(prepared.counts)
    prepared.normalized_features = (
        (features - prepared.calibration_mean[:, None])
        / prepared.calibration_effective_std[:, None]
    ).astype(np.float32)
    evidence = preprocessing_evidence(prepared, reference)
    evidence["numeric_reproducibility"] = {
        "scalers": "canonical_same_fold_Midsize_after_recomputation_check",
        "recomputation_rtol": 5e-7,
        "recomputation_atol": 5e-8,
        "statistics": roundoff,
    }
    return prepared, evidence


def default_gui_root():
    for path in (
        REPO.parent / "BCI-STM32-Plot/data/ai_device_sessions",
        REPO.parent / "STM32/BCI-STM32-Plot/data/ai_device_sessions",
    ):
        if all(
            (path / f"{name}.npz").is_file() for name in session_data.SESSION_BY_NAME
        ):
            return path.resolve()
    return REPO.parent / "BCI-STM32-Plot/data/ai_device_sessions"


def configure_paths(data_root, gui_root, cache_root):
    session_data.RAW_ROOT = data_root / "raw/indy_loco"
    session_data.PROCESSED_ROOT = data_root / "processed/indy_loco"
    session_data.CACHE_DIR = cache_root
    protocol.GUI_ROOT = gui_root


def source_path(session, indy_cache_root):
    spec = session_data.SESSION_BY_NAME[session]
    if spec.subject == "loco":
        return session_data.processed_loco_path(spec)
    # Reuse the saved Phase-13 input, read-only. No raw-data reinterpretation.
    old_cache = indy_cache_root / f"{session}_4ms.npz"
    return old_cache if old_cache.is_file() else session_data.raw_indy_path(spec)


def load_session(session, indy_cache_root):
    spec = session_data.SESSION_BY_NAME[session]
    path = source_path(session, indy_cache_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing session source: {path}; set --data-root / --indy-cache-root"
        )
    if path.suffix == ".mat":
        session_data.validate_source(spec, checksum=True)
        session_data.indy_cache_path(spec).parent.mkdir(parents=True, exist_ok=True)
        data = session_data.load_session(spec)
    else:
        with np.load(path, allow_pickle=False) as archive:
            if (
                str(archive["session"].item()) != session
                or str(archive["source_md5"].item()) != spec.source_md5
            ):
                raise ValueError(f"Session/source metadata mismatch: {path}")
            presence = archive["spike_presence"]
            channels = 96 if spec.subject == "indy" else 192
            if (
                presence.ndim != 2
                or presence.shape[0] != channels
                or not np.isin(presence, [0, 1]).all()
            ):
                raise ValueError(f"Invalid 4-ms spike presence: {path}")
            velocity = archive["velocity_per_sample"].astype(np.float32)
            if (
                velocity.shape != (presence.shape[1], 2)
                or not np.isfinite(velocity).all()
            ):
                raise ValueError(f"Invalid velocity: {path}")
            data = session_data.SessionData(
                spec=spec,
                spike_presence=presence.astype(np.uint8),
                velocity=velocity,
                reach_bounds=session_data.validate_complete_reaches(
                    spec, archive["reach_bounds"]
                ).copy(),
                channel_names=archive["channel_names"].copy(),
            )
    counts, velocity = session_data.aggregate_40ms(data)
    protocol.verify_gui_arrays(session, counts, velocity)
    return data
