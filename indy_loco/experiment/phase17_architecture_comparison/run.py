"""Common runner. Three train_*.py files fix the architecture independently."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch

from indy_loco.experiment.phase17_architecture_comparison import (
    data_contract as contract,
)
from indy_loco.experiment.phase17_architecture_comparison.models import (
    MODEL_NAMES,
    build_model,
    capacity_report,
)
from indy_loco.experiment.phase17_architecture_comparison.training import (
    TRAINING,
    fit_model,
    verify_recipe,
)

HERE = Path(__file__).resolve().parent
protocol = contract.protocol
session_data = contract.session_data


def parse_args(argv=None, fixed_model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if fixed_model is None:
        parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    else:
        parser.set_defaults(model=fixed_model)
    parser.add_argument(
        "--session", action="append", choices=list(session_data.SESSION_BY_NAME)
    )
    parser.add_argument("--fold", action="append", type=int, choices=range(1, 6))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output-name", help="Single directory name inside Phase17/results"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse verified completed checkpoints; restart a fold interrupted during fitting",
    )
    parser.add_argument("--data-root", type=Path, default=contract.INDY / "data")
    parser.add_argument("--gui-root", type=Path, default=contract.default_gui_root())
    parser.add_argument(
        "--indy-cache-root",
        type=Path,
        default=contract.REFERENCE.parent / ".cache/session_inputs",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Shapes, causality, gradients and reference metadata; no fitting",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Also verify all selected real-data folds; no fitting",
    )
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.resume and (args.validate_only or args.dry_run):
        parser.error("Do not combine --resume with a no-training check")
    return args


def model_preflight(name, device):
    protocol.seed_everything(43, mps=device.type == "mps")
    model = build_model(name).to(device).eval()
    inputs = torch.randn(2, 192, 50, device=device)
    modified = inputs.clone()
    modified[:, :, 30:] += 10
    with torch.no_grad():
        first, second = model(inputs), model(modified)
        if first.shape != (2, 50, 2) or not torch.isfinite(first).all():
            raise ValueError("Model shape/finite-output preflight failed")
        torch.testing.assert_close(first[:, :30], second[:, :30], rtol=1e-5, atol=1e-6)
    # Backward only; no optimizer object, no weight update.
    model.train()
    model(inputs)[:, -1].square().mean().backward()
    if any(
        p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()
    ):
        raise ValueError("Model gradient preflight failed")


def code_fingerprints():
    files = [
        *HERE.glob("*.py"),
        contract.FROZEN / "protocol.py",
        contract.FROZEN / "session_data.py",
        contract.FROZEN / "model.py",
        contract.FROZEN / "protocol_lock.json",
    ]
    # Normalized newlines permit checkout on Windows or Unix without false drift.
    return {
        str(p.relative_to(REPO)).replace("\\", "/"): hashlib.sha256(
            p.read_text(encoding="utf-8").encode()
        ).hexdigest()
        for p in sorted(files)
    }


def make_config(args, sessions, folds, include_data):
    config = {
        "phase": "phase17_architecture_comparison",
        "schema_version": 1,
        "model": args.model,
        "capacity": capacity_report(args.model),
        "training": asdict(TRAINING),
        "sessions": sessions,
        "folds": folds,
        "device": args.device,
        "threads": args.threads,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
        },
        "code_sha256": code_fingerprints(),
        "references": {},
        "inputs": {},
        "selection": "minimum_validation_normalized_MSE; test only after all requested checkpoints saved",
        "comparison_note": "Scratch fixed-recipe model comparison; historical Midsize was warm-started. LR group mapping is architecture-specific.",
    }
    for session in sessions:
        for fold in folds:
            contract.load_reference(session, fold)
            config["references"][f"{session}|{fold}"] = protocol.sha256_file(
                contract.reference_path(session, fold)
            )
        if include_data:
            path = contract.source_path(session, args.indy_cache_root)
            gui = args.gui_root / f"{session}.npz"
            for item in (path, gui):
                if not item.is_file():
                    raise FileNotFoundError(
                        f"Missing {item}; set --data-root / --indy-cache-root / --gui-root"
                    )
            config["inputs"][session] = {
                "source_path": str(path.resolve()),
                "source_sha256": protocol.sha256_file(path),
                "gui_path": str(gui.resolve()),
                "gui_sha256": protocol.sha256_file(gui),
            }
    return config


def signature_for(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


@contextmanager
def exclusive_run(output):
    """OS releases the advisory lock on crash; different model directories can coexist."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another process is using {output}") from error
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def validate_saved(checkpoint, signature, evidence, session, fold):
    if (
        checkpoint.get("signature") != signature
        or checkpoint.get("preprocessing_evidence") != evidence
    ):
        raise ValueError("Resume configuration or preprocessed arrays changed")
    if checkpoint.get("session") != session or checkpoint.get("fold") != fold:
        raise ValueError("Resume checkpoint identity mismatch")


def create_checkpoint(
    name, session, fold, fold_data, reference, evidence, signature, output, device
):
    protocol.seed_everything(43, mps=device.type == "mps")
    model = build_model(name).to(device)
    fit = fit_model(
        model,
        name,
        session,
        fold,
        fold_data,
        device,
        output / "epochs" / f"{session}_fold{fold}.json",
    )
    metadata_keys = (
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
    )
    return {
        **{key: reference[key] for key in metadata_keys},
        **fit,
        "session": session,
        "subject": session_data.SESSION_BY_NAME[session].subject,
        "fold": fold,
        "seed": 43,
        "model": name,
        "initialization": "scratch",
        "status": "experimental_not_promoted",
        "created_at_utc": protocol.utc_now(),
        "signature": signature,
        "preprocessing_evidence": evidence,
        "split_indices": {
            key: getattr(fold_data, key).copy()
            for key in (
                "train_reaches",
                "validation_reaches",
                "test_reaches",
                "train_bins",
                "validation_bins",
                "test_bins",
            )
        },
        "capacity": capacity_report(name),
        "training": asdict(TRAINING),
        "reference_checkpoint_used_for_metadata_only": True,
        "selection_policy": "minimum_validation_normalized_MSE",
        "test_evaluated_during_training": False,
    }


def save_training_checkpoint(path, checkpoint):
    protocol.save_checkpoint_atomic(path, checkpoint)
    protocol.write_json_atomic(
        path.with_suffix(".sha256.json"), {"sha256": protocol.sha256_file(path)}
    )


def load_training_checkpoint(path):
    expected = json.loads(path.with_suffix(".sha256.json").read_text(encoding="utf-8"))[
        "sha256"
    ]
    if protocol.sha256_file(path) != expected:
        raise ValueError(f"Saved checkpoint SHA mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate_checkpoint(checkpoint, fold_data, checkpoint_path, output, device):
    session, fold, name = checkpoint["session"], checkpoint["fold"], checkpoint["model"]
    prediction_path = output / "predictions" / f"{session}_fold{fold}.npz"
    checkpoint_hash = protocol.sha256_file(checkpoint_path)
    if prediction_path.exists():
        with np.load(prediction_path, allow_pickle=False) as stored:
            if str(
                stored["checkpoint_sha256"].item()
            ) != checkpoint_hash or not np.array_equal(
                stored["bins"], fold_data.test_bins
            ):
                raise ValueError("Saved prediction checkpoint/mask mismatch")
            prediction = stored["prediction"].copy()
            elapsed = float(stored["seconds"].item())
    else:
        model = build_model(name).to(device).eval()
        model.load_state_dict(checkpoint["model_state"])
        started = time.perf_counter()
        prediction = protocol.predict_last(
            model,
            fold_data.normalized_features,
            fold_data.test_bins,
            fold_data.target_mean,
            fold_data.target_std,
            device,
            TRAINING.batch_size,
        )
        elapsed = time.perf_counter() - started
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite test prediction")
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        session_data.save_npz_atomic(
            prediction_path,
            prediction=prediction,
            target=fold_data.velocity[fold_data.test_bins],
            bins=fold_data.test_bins,
            checkpoint_sha256=np.asarray(checkpoint_hash),
            seconds=np.asarray(elapsed),
        )
    if (
        prediction.shape != (len(fold_data.test_bins), 2)
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("Invalid saved test prediction")
    score = protocol.metrics(fold_data.velocity[fold_data.test_bins], prediction)
    score["rmse"] = float(np.sqrt(score["mse"]))
    return {
        "session": session,
        "subject": checkpoint["subject"],
        "fold": fold,
        "best_epoch": checkpoint["best_epoch"],
        "best_validation_loss": checkpoint["best_validation_loss"],
        "test": score,
        "training_seconds": checkpoint["training_seconds"],
        "test_prediction_seconds": elapsed,
        "history": checkpoint["history"],
        "checkpoint": checkpoint_path.relative_to(output).as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "predictions": prediction_path.relative_to(output).as_posix(),
        "predictions_sha256": protocol.sha256_file(prediction_path),
        "signature": checkpoint["signature"],
        "preprocessing_evidence": checkpoint["preprocessing_evidence"],
    }


def historical_scores():
    path = contract.REFERENCE / "phase13_round3_folds.csv"
    with path.open(newline="", encoding="utf-8") as source:
        return {
            (row["session"], int(row["fold"])): float(row["retrained_7min_rolling_r2"])
            for row in csv.DictReader(source)
        }


def write_reports(output, config, results):
    historical = historical_scores()
    rows = []
    for result in results:
        reference = historical[(result["session"], result["fold"])]
        rows.append(
            {
                "session": result["session"],
                "subject": result["subject"],
                "fold": result["fold"],
                "best_epoch": result["best_epoch"],
                "validation_loss": result["best_validation_loss"],
                **result["test"],
                "historical_midsize_r2": reference,
                "delta_r2_vs_historical": result["test"]["r2_mean"] - reference,
                "training_seconds": result["training_seconds"],
                "test_prediction_seconds": result["test_prediction_seconds"],
            }
        )
    summary = []
    for label in [*config["sessions"], "indy", "loco", "overall_fold_macro"]:
        selected = [
            row
            for row in rows
            if label in (row["session"], row["subject"], "overall_fold_macro")
        ]
        if not selected:
            continue
        values = np.asarray([row["r2_mean"] for row in selected], dtype=np.float64)
        delta = np.asarray(
            [row["delta_r2_vs_historical"] for row in selected], dtype=np.float64
        )
        summary.append(
            {
                "group": label,
                "folds": len(values),
                "r2_mean": float(values.mean()),
                "r2_std": float(values.std(ddof=1)) if len(values) > 1 else None,
                "worst_fold_r2": float(values.min()),
                "delta_vs_historical_mean": float(delta.mean()),
                "wins_vs_historical": int((delta > 0).sum()),
            }
        )
    complete = len(rows) == len(config["sessions"]) * len(config["folds"])
    metrics = {
        "status": "complete" if complete else "partial",
        "full_30fold": complete and len(rows) == 30,
        "completed_folds": len(rows),
        "config": config,
        "summary": summary,
        "results": results,
    }
    protocol.write_csv(output / "folds.csv", rows)
    protocol.write_csv(output / "summary.csv", summary)
    protocol.write_csv(
        output / "epochs.csv",
        [epoch for result in results for epoch in result["history"]],
    )
    protocol.write_json_atomic(output / "metrics.json", metrics)


def run_training(args, config, signature, output, device):
    # Stage 1: select/save ALL requested folds before opening any test results.
    for session in config["sessions"]:
        data = contract.load_session(session, args.indy_cache_root)
        for fold in config["folds"]:
            reference = contract.load_reference(session, fold)
            prepared, evidence = contract.prepare_verified_fold(data, fold, reference)
            path = output / "checkpoints" / f"{session}_fold{fold}.pt"
            if path.exists():
                saved = load_training_checkpoint(path)
                validate_saved(saved, signature, evidence, session, fold)
                print(f"RESUME checkpoint verified: {session} fold {fold}", flush=True)
            else:
                print(
                    f"FIT {session} fold {fold}: preprocessing matches Midsize",
                    flush=True,
                )
                saved = create_checkpoint(
                    args.model,
                    session,
                    fold,
                    prepared,
                    reference,
                    evidence,
                    signature,
                    output,
                    device,
                )
                save_training_checkpoint(path, saved)
    results = []
    # Stage 2: fixed test evaluation and resumable result receipts.
    for session in config["sessions"]:
        data = contract.load_session(session, args.indy_cache_root)
        for fold in config["folds"]:
            prepared, evidence = contract.prepare_verified_fold(
                data, fold, contract.load_reference(session, fold)
            )
            path = output / "checkpoints" / f"{session}_fold{fold}.pt"
            checkpoint = load_training_checkpoint(path)
            validate_saved(checkpoint, signature, evidence, session, fold)
            receipt = output / "fold_results" / f"{session}_fold{fold}.json"
            if receipt.exists():
                result = json.loads(receipt.read_text(encoding="utf-8"))
                validate_saved(result, signature, evidence, session, fold)
                for key in ("checkpoint", "predictions"):
                    if (
                        protocol.sha256_file(output / result[key])
                        != result[f"{key}_sha256"]
                    ):
                        raise ValueError(f"Saved {key} modified: {session} fold {fold}")
            else:
                result = evaluate_checkpoint(checkpoint, prepared, path, output, device)
                protocol.write_json_atomic(receipt, result)
            results.append(result)
            write_reports(output, config, results)
            print(
                f"TEST {session} fold {fold}: R2={result['test']['r2_mean']:.6f}",
                flush=True,
            )
    print(f"COMPLETE {len(results)} folds: {output}", flush=True)


def main(argv=None, fixed_model=None):
    args = parse_args(argv, fixed_model)
    lock = contract.verify_protocol_lock()
    verify_recipe(lock)
    torch.set_num_threads(args.threads)
    device = protocol.select_device(args.device)
    name = args.output_name or args.model
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("--output-name must be a single safe directory name")
    root = (HERE / "results").resolve()
    output = root / name
    if output.resolve().parent != root or output.is_symlink():
        raise ValueError("Results must stay inside this phase")
    args.data_root = args.data_root.resolve()
    args.gui_root = args.gui_root.resolve()
    args.indy_cache_root = args.indy_cache_root.resolve()
    contract.configure_paths(args.data_root, args.gui_root, root / ".cache" / name)
    sessions = list(dict.fromkeys(args.session or session_data.SESSION_BY_NAME))
    folds = sorted(set(args.fold or range(1, 6)))
    model_preflight(args.model, device)
    config = make_config(args, sessions, folds, include_data=not args.validate_only)
    print(
        json.dumps(
            {
                "model": args.model,
                "capacity": config["capacity"],
                "device": args.device,
                "planned_folds": len(sessions) * len(folds),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.validate_only:
        print(
            "VALIDATED: model forward/backward + reference metadata; zero optimizer steps; no training outputs"
        )
        return
    if args.dry_run:
        for session in sessions:
            data = contract.load_session(session, args.indy_cache_root)
            for fold in folds:
                prepared, evidence = contract.prepare_verified_fold(
                    data, fold, contract.load_reference(session, fold)
                )
                print(
                    f"VERIFIED {session} fold {fold}: train/val/test={len(prepared.train_bins)}/{len(prepared.validation_bins)}/{len(prepared.test_bins)}",
                    flush=True,
                )
        print(
            "DRY RUN PASSED: all selected preprocessing/splits match Midsize; zero optimizer steps"
        )
        return
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite {output}; use --resume or --output-name"
        )
    if args.resume and not (output / "config.json").is_file():
        raise FileNotFoundError("--resume requires a previously started run")
    signature = signature_for(config)
    with exclusive_run(output):
        config_path = output / "config.json"
        if args.resume:
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if signature_for(previous) != signature:
                raise ValueError(
                    "Resume config/code/environment/input changed; use a new output directory"
                )
        else:
            if config_path.exists():
                raise FileExistsError("Another run already created this configuration")
            protocol.write_json_atomic(config_path, config)
        run_training(args, config, signature, output, device)


if __name__ == "__main__":
    main()
