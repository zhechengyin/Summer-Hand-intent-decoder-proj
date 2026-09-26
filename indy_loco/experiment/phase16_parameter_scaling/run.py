#!/usr/bin/env python3
"""Phase 16: modest scaling with fold-matched Midsize weight transfer and a frozen protocol."""

# ruff: noqa: E402  # Repository bootstrap and CUDA configuration precede torch.
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch

from indy_loco.experiment.phase16_parameter_scaling import (
    fast_training,
    protocol,
    session_data,
)
from indy_loco.experiment.phase16_parameter_scaling.model import (
    BASELINE,
    Architecture,
    ScaledTCNGRU,
    parameter_report,
)
from indy_loco.experiment.phase16_parameter_scaling.stopping import StopCriterion
from indy_loco.experiment.phase16_parameter_scaling.transfer import (
    transfer_midsize_weights,
)

HERE = Path(__file__).resolve().parent
INDY = REPO / "indy_loco"
REFERENCE = (
    INDY
    / "experiment/phase13_deployment_validation/results/rolling_retrain/final_30fold"
)


@dataclass(frozen=True)
class Training:
    # Initialization is the user's explicit exception to the Phase-13 recipe.
    init: str = "midsize_transfer"
    train_scope: str = "all"
    epochs: int = 20
    batch_size: int = 128
    patience: int = 6
    weight_decay: float = 0.025
    gradient_clip: float = 1.0


TRAINING = Training()
FAST_TRAINING = replace(TRAINING, patience=StopCriterion().patience)
LEARNING_RATE = 3e-4
ENCODER_LR_SCALE = 0.25


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--session", action="append", choices=list(session_data.SESSION_BY_NAME)
    )
    p.add_argument("--fold", action="append", type=int, choices=range(1, 6))
    p.add_argument(
        "--model",
        choices=("scaled", "baseline"),
        default="scaled",
        help="baseline is the original 86,978-param architecture, also warm-started from its matching Midsize fold",
    )
    p.add_argument("--encoder-width", type=int, default=80)
    p.add_argument("--encoder-kernel-size", type=int, default=3)
    p.add_argument("--encoder-dilations", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument(
        "--encoder-layers",
        type=int,
        help="Optional consistency assertion; must equal number of dilations",
    )
    p.add_argument("--decoder-hidden-size", type=int, default=80)
    p.add_argument("--decoder-layers", type=int, default=1)
    p.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="CPU matches the final Midsize run; other backends are recorded",
    )
    p.add_argument("--output-name", help="New folder inside this phase's results only")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip verified completed folds; restart interrupted fold",
    )
    modes = p.add_mutually_exclusive_group()
    modes.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate code, shapes and checkpoint metadata; no training",
    )
    modes.add_argument(
        "--dry-run",
        action="store_true",
        help="Also verify every selected data split/preprocessing; no training",
    )
    return p.parse_args(argv)


def architecture_for(args):
    architecture = (
        BASELINE
        if args.model == "baseline"
        else Architecture(
            args.encoder_width,
            args.encoder_kernel_size,
            tuple(args.encoder_dilations),
            args.decoder_hidden_size,
            args.decoder_layers,
        )
    )
    if (
        args.encoder_layers is not None
        and args.encoder_layers != architecture.encoder_layers
    ):
        raise ValueError("--encoder-layers must equal the number of encoder dilations")
    if args.model == "baseline" and (
        args.encoder_width != 80
        or args.encoder_kernel_size != 3
        or args.encoder_dilations != [1, 2, 4, 8]
        or args.decoder_hidden_size != 80
        or args.decoder_layers != 1
    ):
        raise ValueError("Do not combine --model baseline with size overrides")
    return architecture


def node_hash(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def check_protocol_lock():
    lock = json.loads((HERE / "protocol_lock.json").read_text())
    for filename, key in (
        ("session_data.py", "session_data_nodes"),
        ("protocol.py", "protocol_nodes"),
    ):
        tree = ast.parse((HERE / filename).read_text())
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
        for name, expected in lock[key].items():
            if name not in nodes or node_hash(nodes[name]) != expected:
                raise ValueError(f"Frozen protocol changed: {filename}:{name}")
        if filename == "protocol.py":
            # fit_model has a docstring/import/seed prefix and a return suffix.
            body = nodes["fit_model"].body[3:-1]
            digest = hashlib.sha256(
                "\n".join(ast.dump(n, include_attributes=False) for n in body).encode()
            ).hexdigest()
            if digest != lock["training_core_sha256"]:
                raise ValueError("Frozen Phase-13 training loop changed")
    expected = lock["training"]
    for key in (
        "train_scope",
        "epochs",
        "batch_size",
        "patience",
        "weight_decay",
        "gradient_clip",
    ):
        if getattr(TRAINING, key) != expected[key]:
            raise ValueError(f"Frozen training setting changed: {key}")
    if (
        TRAINING.init != "midsize_transfer"
        or LEARNING_RATE != expected["learning_rate_gru_head"]
        or ENCODER_LR_SCALE != expected["encoder_lr_scale"]
    ):
        raise ValueError("Initialization or learning-rate contract changed")
    # Apart from the stopping predicate, preserve every original fitting statement.
    original = nodes["fit_model"].body[3:-1]
    fast_tree = ast.parse((HERE / "fast_training.py").read_text())
    fast_function = next(
        n
        for n in fast_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "fit_model"
    )
    actual = copy.deepcopy(fast_function.body[4:-1])
    original_loop = next(
        n
        for n in original
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "epoch"
    )
    fast_loop = next(
        n
        for n in actual
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "epoch"
    )
    fast_loop.body[-1] = copy.deepcopy(original_loop.body[-1])
    if [ast.dump(n) for n in actual] != [ast.dump(n) for n in original]:
        raise ValueError(
            "Fast training changed more than the authorized stopping predicate"
        )
    if StopCriterion().max_epochs != TRAINING.epochs:
        raise ValueError(
            "Stopping cap must preserve the original epoch/scheduler budget"
        )
    return lock


def canonical_path(session, fold):
    directory = INDY / "models/midsize" / session
    manifest = json.loads((directory / "manifest.json").read_text())
    row = next(r for r in manifest["model"]["checkpoints"] if r["fold"] == fold)
    path = directory / row["file"]
    if protocol.sha256_file(path) != row["sha256"]:
        raise ValueError(f"Canonical checkpoint checksum mismatch: {path}")
    return path


def validate_metadata(session, fold):
    path = canonical_path(session, fold)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
    if any(checkpoint.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Not a canonical final Midsize checkpoint: {path}")
    policy = checkpoint["deployment_policy"]
    if (
        policy["calibration_bins"] != 10500
        or policy["window_bins"] != 50
        or policy["ewma_alpha"] != 0.1
    ):
        raise ValueError("Canonical preprocessing contract changed")
    return checkpoint


def verify_preprocessing(fold_data, reference):
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
            raise ValueError(f"Preprocessing differs from final Midsize: {field}")
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
            raise ValueError(f"Reach count changed: {split}")
    arrays = {}
    for name in (
        "channels",
        "normalized_features",
        "target_mean",
        "target_std",
        "train_reaches",
        "validation_reaches",
        "test_reaches",
        "train_bins",
        "validation_bins",
        "test_bins",
    ):
        array = np.ascontiguousarray(getattr(fold_data, name))
        arrays[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    return arrays


def load_session(session):
    spec = session_data.SESSION_BY_NAME[session]
    session_data.validate_source(spec, checksum=False)
    # Reuse validated old caches by copying; never rewrite old experiment files.
    old_cache = REFERENCE.parent / ".cache/session_inputs" / f"{session}_4ms.npz"
    new_cache = session_data.indy_cache_path(spec)
    if spec.subject == "indy" and not new_cache.exists() and old_cache.is_file():
        new_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_cache, new_cache)
    data = session_data.load_session(spec)
    counts, velocity = session_data.aggregate_40ms(data)
    protocol.verify_gui_arrays(session, counts, velocity)
    return data


def train_one(
    data, fold, fold_data, architecture, device, output, signature, evidence, reference
):
    protocol.seed_everything(43, mps=device.type == "mps")
    model = ScaledTCNGRU(architecture)
    transfer = transfer_midsize_weights(model, reference["model_state"])
    model.to(device)
    best_state, best_epoch, history, best_loss = fast_training.fit_model(
        model,
        data,
        fold - 1,
        fold_data,
        FAST_TRAINING,
        device,
        LEARNING_RATE,
        ENCODER_LR_SCALE,
    )
    checkpoint = {
        key: reference[key]
        for key in (
            "session",
            "subject",
            "fold",
            "seed",
            "selected_channel_indices",
            "selected_channel_names",
            "source_channel_count",
            "physical_channel_count",
            "input_feature_count",
            "feature_mean",
            "feature_std",
            "feature_std_floor",
            "calibration_local_std",
            "target_mean",
            "target_std",
            "deployment_policy",
            "floor_fit",
            "reach_counts",
            "bin_counts_after_calibration",
            "selection_policy",
            "test_evaluated_during_training",
        )
    }
    checkpoint.update(
        purpose="phase16_parameter_scaling",
        status="experimental_not_promoted",
        created_at_utc=protocol.utc_now(),
        model_state=best_state,
        best_epoch=best_epoch,
        initialization=TRAINING.init,
        weight_transfer=transfer,
        stopping_criterion=StopCriterion().metadata(),
        initialization_checkpoint=str(
            canonical_path(data.spec.name, fold).relative_to(INDY)
        ),
        initialization_checkpoint_sha256=protocol.sha256_file(
            canonical_path(data.spec.name, fold)
        ),
        train_scope="all",
        architecture=architecture.metadata(),
        parameter_count=parameter_report(architecture)["total_parameters"],
        signature=signature,
        preprocessing_evidence=evidence,
        reference_checkpoint_used_for_metadata_only=False,
    )
    path = output / "checkpoints" / f"{data.spec.name}_fold{fold}.pt"
    # Freeze the validation-selected checkpoint BEFORE the only test prediction pass.
    protocol.save_checkpoint_atomic(path, checkpoint)
    score = protocol.evaluate_last(
        model,
        fold_data.normalized_features,
        fold_data.velocity,
        fold_data.test_bins,
        fold_data.target_mean,
        fold_data.target_std,
        device,
        TRAINING.batch_size,
    )
    return {
        "session": data.spec.name,
        "subject": data.spec.subject,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "test": score,
        "checkpoint": str(path.relative_to(output)),
        "checkpoint_sha256": protocol.sha256_file(path),
        "history": history,
        "preprocessing_evidence": evidence,
    }


def save_reports(output, results, config, complete):
    rows = [
        {
            "session": r["session"],
            "fold": r["fold"],
            "best_epoch": r["best_epoch"],
            "validation_loss": r["best_validation_loss"],
            **r["test"],
        }
        for r in results
    ]
    summary = []
    for session in [*config["sessions"], "overall_fold_macro"]:
        selected = (
            rows
            if session == "overall_fold_macro"
            else [r for r in rows if r["session"] == session]
        )
        if selected:
            values = np.asarray([r["r2_mean"] for r in selected], dtype=np.float64)
            summary.append(
                {
                    "session": session,
                    "folds": len(values),
                    "r2_mean": float(values.mean()),
                    "r2_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
            )
    protocol.write_csv(output / "folds.csv", rows)
    protocol.write_csv(
        output / "epochs.csv", [e for r in results for e in r["history"]]
    )
    protocol.write_csv(output / "summary.csv", summary)
    protocol.write_json_atomic(
        output / "metrics.json",
        {
            "status": "complete" if complete else "partial",
            "config": config,
            "completed_folds": len(rows),
            "summary": summary,
            "results": results,
            "full_30fold": complete and len(rows) == 30,
            "caveat": "Both scaled and baseline controls warm-start from the corresponding final Phase-13 fold. Additional optimization differs from the published baseline; never select architecture by test score.",
        },
    )


def main(argv=None):
    args = parse_args(argv)
    check_protocol_lock()
    architecture = architecture_for(args)
    sessions = list(dict.fromkeys(args.session or session_data.SESSION_BY_NAME))
    folds = sorted(set(args.fold or range(1, 6)))
    torch.set_num_threads(4)
    device = protocol.select_device(args.device)
    config = {
        "phase": "phase16_parameter_scaling",
        "architecture": architecture.metadata(),
        "parameters": parameter_report(architecture),
        "training": vars(FAST_TRAINING),
        "stopping_criterion": StopCriterion().metadata(),
        "learning_rate_gru_head": LEARNING_RATE,
        "encoder_lr_scale": ENCODER_LR_SCALE,
        "device": device.type,
        "threads": 4,
        "sessions": sessions,
        "folds": folds,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "protocol_lock_sha256": protocol.sha256_file(HERE / "protocol_lock.json"),
        "code_sha256": {
            p.name: protocol.sha256_file(p) for p in sorted(HERE.glob("*.py"))
        },
        "reference_checkpoints": {},
        "transfer_preflight": {},
        "gui_data_sha256": {},
        "authorized_protocol_changes": "User revised initialization to allow matching final Midsize fold weight transfer for modest scaling. New parameters keep default random initialization; only early stopping additionally changes to min 4 epochs / patience 3 / 0.5% relative validation improvement; all other hyperparameters stay frozen.",
    }
    for session in sessions:
        config["gui_data_sha256"][session] = protocol.sha256_file(
            protocol.GUI_ROOT / f"{session}.npz"
        )
        for fold in folds:
            reference = validate_metadata(session, fold)
            candidate = ScaledTCNGRU(architecture)
            config["transfer_preflight"][f"{session}|{fold}"] = (
                transfer_midsize_weights(candidate, reference["model_state"])
            )
            with torch.inference_mode():
                if not torch.isfinite(candidate.eval()(torch.zeros(1, 192, 50))).all():
                    raise ValueError(
                        f"Nonfinite transferred model: {session} fold {fold}"
                    )
            del candidate
            config["reference_checkpoints"][f"{session}|{fold}"] = protocol.sha256_file(
                canonical_path(session, fold)
            )
    with torch.inference_mode():
        model = ScaledTCNGRU(architecture).eval()
        if model(torch.zeros(2, 192, 50)).shape != (2, 50, 2):
            raise ValueError("Output contract changed")
    del model
    print(json.dumps(config, indent=2), flush=True)
    if args.validate_only:
        print("VALIDATED: no training, optimizer steps or result files created")
        return
    if architecture.metadata()["encoder_receptive_field_bins"] > 50:
        print(
            "Encoder receptive field exceeds 50; earlier context remains zero padding. Input window is unchanged."
        )
    default_name = (
        "fast_transfer_baseline" if args.model == "baseline" else "fast_transfer_scaled"
    )
    name = args.output_name or default_name
    if (
        not name
        or name in (".", "..")
        or Path(name).name != name
        or "/" in name
        or "\\" in name
    ):
        raise ValueError("--output-name must be a single directory name")
    output = HERE / "results" / name
    if output.is_symlink():
        raise ValueError("Result directory must not be a symlink")
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    state_path = output / "state.json"
    state = {"signature": signature, "completed": {}}
    if not args.dry_run:
        if args.resume:
            state = json.loads(state_path.read_text())
            if state["signature"] != signature:
                raise ValueError(
                    "Resume configuration/code/assets changed; use a new output directory"
                )
        elif output.exists():
            raise FileExistsError(
                f"Refusing to overwrite {output}; use --resume or a new --output-name"
            )
        else:
            protocol.write_json_atomic(output / "config.json", config)
            protocol.write_json_atomic(state_path, state)
    for session in sessions:
        data = load_session(session)
        for fold in folds:
            fold_data = protocol.prepare_fold(data, fold - 1)
            reference = validate_metadata(session, fold)
            evidence = verify_preprocessing(fold_data, reference)
            key = f"{session}|{fold}"
            print(
                f"VERIFIED {key}: train/val/test={len(fold_data.train_bins)}/{len(fold_data.validation_bins)}/{len(fold_data.test_bins)}",
                flush=True,
            )
            if args.dry_run:
                continue
            if key in state["completed"]:
                saved = state["completed"][key]
                if (
                    evidence != saved["preprocessing_evidence"]
                    or protocol.sha256_file(output / saved["checkpoint"])
                    != saved["checkpoint_sha256"]
                ):
                    raise ValueError(f"Resume data/checkpoint changed: {key}")
                print(f"resume: verified completed {key}")
                continue
            state["completed"][key] = train_one(
                data,
                fold,
                fold_data,
                architecture,
                device,
                output,
                signature,
                evidence,
                reference,
            )
            protocol.write_json_atomic(state_path, state)
            save_reports(output, list(state["completed"].values()), config, False)
    if args.dry_run:
        print(
            "DRY RUN PASSED: all selected folds match final Midsize; no training or scores"
        )
        return
    save_reports(output, list(state["completed"].values()), config, True)
    print(f"Completed {len(state['completed'])} folds: {output}")


if __name__ == "__main__":
    main()
