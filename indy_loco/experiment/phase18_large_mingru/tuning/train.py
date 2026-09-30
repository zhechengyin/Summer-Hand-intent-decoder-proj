"""Tune head regularization without changing minGRU capacity or training duration."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from indy_loco.experiment.phase17_architecture_comparison import (
    data_contract as contract,
)
from indy_loco.experiment.phase17_architecture_comparison import run as original
from indy_loco.experiment.phase18_large_mingru.tuning import model as models
from indy_loco.experiment.phase18_large_mingru.tuning import plan

protocol = contract.protocol
PHASE = HERE.parent
RESULTS = PHASE.parent / "phase17_architecture_comparison/results"
OUTPUT = PHASE / "results/regularization_v2"
PREVIOUS = PHASE / "results/large_mingru_v1"
SESSIONS = tuple(contract.session_data.SESSION_BY_NAME)
CACHE_ROOT = contract.REFERENCE.parent / ".cache/session_inputs"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prior_validation_provenance():
    """Fingerprint the completed training evidence, without reading test scores."""
    source_config = read_json(PREVIOUS / "config.json")
    for relative, expected in source_config["code_sha256"].items():
        actual = hashlib.sha256(
            (REPO / relative).read_text(encoding="utf-8").encode()
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen prior experiment code changed: {relative}")
    histories = sorted((PREVIOUS / "runs").glob("*/t*/seed*/epochs/*.json"))
    if len(histories) != 84:
        raise ValueError("Expected all 84 prior learning curves before tuning")
    return {
        "config_sha256": protocol.sha256_file(PREVIOUS / "config.json"),
        "validation_selection_sha256": protocol.sha256_file(
            PREVIOUS / "final_selection.json"
        ),
        "epoch_history_sha256": {
            path.relative_to(PREVIOUS).as_posix(): protocol.sha256_file(path)
            for path in histories
        },
        "test_scores_used_for_tuning": False,
    }


def build_config(device, threads):
    args = SimpleNamespace(
        model="mingru",
        device=device,
        threads=threads,
        indy_cache_root=CACHE_ROOT,
        gui_root=contract.default_gui_root(),
    )
    base = original.make_config(
        args, list(SESSIONS), list(plan.FOLDS), include_data=True
    )
    code = base["code_sha256"]
    for path in sorted([*PHASE.glob("*.py"), *HERE.glob("*.py")]):
        code[path.relative_to(REPO).as_posix()] = hashlib.sha256(
            path.read_text(encoding="utf-8").encode()
        ).hexdigest()
    return {
        "phase": "phase18_large_mingru",
        "extension": "regularization_v2",
        "models": list(plan.MODELS),
        "trials": plan.TRIALS,
        "fixed": plan.FIXED,
        "seeds": list(plan.SEEDS),
        "sessions": list(SESSIONS),
        "folds": list(plan.FOLDS),
        "screen_folds": list(plan.SCREEN_FOLDS),
        "top_k": plan.TOP_K,
        "fit_budget": plan.expected_fits(),
        "device": device,
        "threads": threads,
        "environment": {
            **base["environment"],
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "inputs": base["inputs"],
        "references": base["references"],
        "prior_validation_provenance": prior_validation_provenance(),
        "training_diagnostic": "Evaluation-mode loss on at most 1024 deterministic training bins each epoch; diagnostic only, never checkpoint/configuration selection.",
        "code_sha256": code,
        "capacity": {
            name: {
                key: value
                for key, value in models.capacity_report().items()
                if key != "parameter_groups"
            }
            for name in plan.MODELS
        },
        "selection": "Validation only; eight trials on six fixed screening folds; top two on all 30 folds; freeze one winner; test its 30 folds only after all 96 fits. Seed 43 only. Independent zero-state windows, no persistent streaming.",
        "limitations": "Exposed historical test; overlapping cross-validation folds and global HPO are not nested independent test estimation. Correlated folds; six sessions. Historical Midsize warm-started. Large model trains from scratch; board memory placement and latency are not established by GPU training. FP32, no quantization or persistent streaming.",
    }


def create_model(name, trial):
    if name not in plan.MODELS:
        raise ValueError(f"Unknown large model: {name}")
    model = models.LargeMinGRU(
        channel_dropout=trial["channel_dropout"],
        dropout=trial["dropout"],
        head_dropout=trial["head_dropout"],
    )
    if sum(p.numel() for p in model.parameters()) != 1_211_906:
        raise ValueError("Large minGRU parameter capacity changed")
    return model


def optimizer_groups(model, name, trial):
    if name not in plan.MODELS:
        raise ValueError(f"Unknown model: {name}")
    return models.parameter_groups(
        model,
        learning_rate=trial["learning_rate"],
        encoder_lr_scale=trial["stem_lr_scale"],
        weight_decay=trial["weight_decay"],
        head_weight_decay=trial["head_weight_decay"],
    )


def verify_trained_arithmetic(model, data, device):
    """Check deployment recurrence on validation inputs, never held-out targets."""
    bins = data.validation_bins[:32]
    if len(bins) == 0:
        raise ValueError("No validation inputs for trained arithmetic check")
    inputs = torch.from_numpy(
        protocol.rolling_batch(data.normalized_features, bins)
    ).to(device)
    model.eval()
    with torch.inference_mode():
        parallel = model(inputs)
        sequential = model.forward_sequential(inputs)
        reference = model.forward_reference(inputs)[:, -1:]
        torch.testing.assert_close(parallel, sequential, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(parallel, reference, rtol=2e-4, atol=2e-5)
    return {
        "validation_windows": int(len(bins)),
        "max_abs_parallel_vs_sequential": float((parallel - sequential).abs().max()),
        "max_abs_pruned_vs_full": float((parallel - reference).abs().max()),
        "rtol": 2e-4,
        "atol": 2e-5,
        "persistent_state": False,
    }


def evaluate_training_probe(model, data, device):
    """Comparable dropout-off loss on a fixed, training-only diagnostic sample."""
    positions = np.linspace(
        0, len(data.train_bins) - 1, min(1024, len(data.train_bins)), dtype=int
    )
    bins = data.train_bins[positions]
    if len(bins) == 0:
        raise ValueError("Empty training split")
    prediction = protocol.predict_last(
        model,
        data.normalized_features,
        bins,
        data.target_mean,
        data.target_std,
        device,
        plan.FIXED["batch_size"],
    )
    score = protocol.metrics(data.velocity[bins], prediction)
    score["normalized_loss"] = float(
        np.mean(((data.velocity[bins] - prediction) / data.target_std) ** 2)
    )
    if not all(
        math.isfinite(float(score[key])) for key in ("normalized_loss", "r2_mean")
    ):
        raise FloatingPointError("Nonfinite training diagnostic")
    return score


def fit(model, name, trial, seed, data, device, epoch_path, label):
    """Same window/loss/sampling as Phase17; only explicit sweep settings differ."""
    fixed = plan.FIXED
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, name, trial), weight_decay=trial["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, fixed["epochs"])
    rng = np.random.default_rng(seed)
    targets = ((data.velocity - data.target_mean) / data.target_std).astype(np.float32)
    best_state, best_score, best_epoch, stale = None, None, 0, 0
    best_loss = math.inf
    history = []
    started = time.perf_counter()
    for epoch in range(1, fixed["epochs"] + 1):
        model.train()
        order = rng.permutation(data.train_bins)
        error_sum, count, gradient_max = 0.0, 0, 0.0
        for left in range(0, len(order), fixed["batch_size"]):
            bins = order[left : left + fixed["batch_size"]]
            inputs = torch.from_numpy(
                protocol.rolling_batch(data.normalized_features, bins)
            ).to(device)
            target = torch.from_numpy(targets[bins]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)[:, -1]
            loss = (prediction - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite training loss: {label} epoch {epoch}"
                )
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), fixed["gradient_clip"], error_if_nonfinite=True
                )
            )
            optimizer.step()
            error_sum += float(loss.detach()) * len(bins)
            count += len(bins)
            gradient_max = max(gradient_max, gradient)
        score = protocol.evaluate_last(
            model,
            data.normalized_features,
            data.velocity,
            data.validation_bins,
            data.target_mean,
            data.target_std,
            device,
            fixed["batch_size"],
        )
        if not all(
            math.isfinite(float(score[k])) for k in ("normalized_loss", "r2_mean")
        ):
            raise FloatingPointError(f"Nonfinite validation: {label}")
        train_probe = evaluate_training_probe(model, data, device)
        improved = score["normalized_loss"] < best_loss
        if improved:
            best_loss, best_epoch, best_score, stale = (
                score["normalized_loss"],
                epoch,
                score,
                0,
            )
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "optimization_loss": error_sum / count,
                "validation": score,
                "training_eval_probe": train_probe,
                "best": improved,
                "temporal_lr": optimizer.param_groups[0]["lr"],
                "stem_lr": optimizer.param_groups[1]["lr"],
                "head_lr": optimizer.param_groups[2]["lr"],
                "head_weight_decay": optimizer.param_groups[2]["weight_decay"],
                "gradient_max_before_clip": gradient_max,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        protocol.write_json_atomic(epoch_path, history)
        print(
            f"{label} epoch {epoch:02d}/{fixed['epochs']} train={error_sum / count:.5f} val={score['normalized_loss']:.5f} val_R2={score['r2_mean']:+.5f}"
            + (" *best*" if improved else ""),
            flush=True,
        )
        scheduler.step()
        if stale >= fixed["patience"]:
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    arithmetic_check = verify_trained_arithmetic(model, data, device)
    return {
        "model_state": best_state,
        "validation": best_score,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "training_seconds": time.perf_counter() - started,
        "arithmetic_check": arithmetic_check,
    }


@lru_cache(maxsize=1)
def load_data(session):
    return contract.load_session(session, CACHE_ROOT)


@lru_cache(maxsize=1)
def prepare(session, fold):
    data, evidence = contract.prepare_verified_fold(
        load_data(session), fold, contract.load_reference(session, fold)
    )
    # Check exact preprocessing and split evidence against BOTH completed candidates.
    for name in ("mingru", "mamba2"):
        path = RESULTS / name / "fold_results" / f"{session}_fold{fold}.json"
        old = read_json(path)
        if evidence != old["preprocessing_evidence"]:
            raise ValueError(
                f"Preprocessing drift against original {name}: {session}/{fold}"
            )
    return data, evidence


def checkpoint_path(name, trial, seed, session, fold):
    return (
        OUTPUT
        / "runs"
        / name
        / trial
        / f"seed{seed}"
        / "checkpoints"
        / f"{session}_fold{fold}.pt"
    )


def identity(name, trial, seed, session, fold, signature):
    return {
        "model": name,
        "trial": trial,
        "seed": seed,
        "session": session,
        "subject": contract.session_data.SESSION_BY_NAME[session].subject,
        "fold": fold,
        "signature": signature,
    }


def check_saved(saved, expected, evidence):
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved run identity/configuration mismatch")
    if saved.get("preprocessing_evidence") != evidence:
        raise ValueError("Saved preprocessed arrays changed")
    if saved.get("test_evaluated_during_training") is not False:
        raise ValueError("Training checkpoint lacks test isolation declaration")
    score = saved["validation"]
    if not all(math.isfinite(float(score[k])) for k in ("normalized_loss", "r2_mean")):
        raise ValueError("Invalid saved validation score")


def training_receipt(saved, path):
    keys = (
        "model",
        "trial",
        "seed",
        "session",
        "subject",
        "fold",
        "signature",
        "preprocessing_evidence",
        "validation",
        "best_epoch",
        "training_seconds",
        "test_evaluated_during_training",
        "arithmetic_check",
    )
    return {
        **{key: saved[key] for key in keys},
        "checkpoint": path.relative_to(OUTPUT).as_posix(),
        "checkpoint_sha256": protocol.sha256_file(path),
    }


class StopRequested(Exception):
    pass


class Sweep:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.signature = digest(config)
        self.verified = set()

    def progress(self, stage, current=None, **extra):
        protocol.write_json_atomic(
            OUTPUT / "progress.json",
            {
                "status": "running",
                "stage": stage,
                "current": current,
                "verified_fits_this_invocation": len(self.verified),
                "planned_fits": plan.expected_fits(),
                "updated_at_utc": protocol.utc_now(),
                "pid": os.getpid(),
                **extra,
            },
        )

    def assert_unchanged(self):
        if (
            digest(build_config(self.config["device"], self.config["threads"]))
            != self.signature
        ):
            raise ValueError(
                "Code, input, environment or reference changed during sweep"
            )

    def ensure_fit(self, name, trial, seed, session, fold, stage):
        if (OUTPUT / "STOP_AFTER_FOLD").exists():
            raise StopRequested(
                "STOP_AFTER_FOLD requested; completed checkpoints retained"
            )
        label = f"{name}/{trial}/seed{seed}/{session}/fold{fold}"
        self.progress(stage, label)
        data, evidence = prepare(session, fold)
        expected = identity(name, trial, seed, session, fold, self.signature)
        path = checkpoint_path(name, trial, seed, session, fold)
        if path.exists():
            saved = original.load_training_checkpoint(path)
            check_saved(saved, expected, evidence)
            print(f"RESUME verified {label}", flush=True)
        else:
            protocol.seed_everything(seed, mps=False)
            model = create_model(name, plan.TRIALS[trial]).to(self.device)
            epoch_path = path.parent.parent / "epochs" / f"{session}_fold{fold}.json"
            fitted = fit(
                model,
                name,
                plan.TRIALS[trial],
                seed,
                data,
                self.device,
                epoch_path,
                label,
            )
            saved = {
                **expected,
                **fitted,
                "hyperparameters": plan.TRIALS[trial],
                "training": plan.FIXED,
                "preprocessing_evidence": evidence,
                "test_evaluated_during_training": False,
                "split_indices": {
                    key: getattr(data, key).copy()
                    for key in (
                        "train_bins",
                        "validation_bins",
                        "test_bins",
                        "train_reaches",
                        "validation_reaches",
                        "test_reaches",
                    )
                },
                "scalers": {
                    key: getattr(data, key).copy()
                    for key in (
                        "target_mean",
                        "target_std",
                        "calibration_mean",
                        "calibration_effective_std",
                        "channels",
                    )
                },
                "created_at_utc": protocol.utc_now(),
            }
            original.save_training_checkpoint(path, saved)
            del model
        receipt = training_receipt(saved, path)
        receipt_path = path.with_suffix(".validation.json")
        if receipt_path.exists() and read_json(receipt_path) != receipt:
            raise ValueError(f"Validation receipt was modified: {receipt_path}")
        protocol.write_json_atomic(receipt_path, receipt)
        self.verified.add(label)
        return receipt

    def collect(self, name, trials, folds, seeds=(43,)):
        rows = []
        for trial in trials:
            for seed in seeds:
                for session in SESSIONS:
                    for fold in folds:
                        path = checkpoint_path(name, trial, seed, session, fold)
                        row = read_json(path.with_suffix(".validation.json"))
                        if any(
                            row.get(k) != v
                            for k, v in identity(
                                name, trial, seed, session, fold, self.signature
                            ).items()
                        ):
                            raise ValueError("Validation receipt identity mismatch")
                        if protocol.sha256_file(path) != row["checkpoint_sha256"]:
                            raise ValueError("Validation checkpoint hash mismatch")
                        saved = original.load_training_checkpoint(path)
                        if training_receipt(saved, path) != row:
                            raise ValueError(
                                "Validation receipt differs from checkpoint"
                            )
                        rows.append(row)
        return rows

    def freeze(self, filename, payload):
        path = OUTPUT / filename
        if path.exists() and read_json(path) != payload:
            raise ValueError(f"Frozen selection changed: {path}")
        protocol.write_json_atomic(path, payload)

    def run(self):
        # Pair model/trial work on the same prepared fold; one fit uses the GPU.
        for session in SESSIONS:
            for fold in plan.SCREEN_FOLDS:
                for trial in plan.TRIALS:
                    for name in plan.MODELS:
                        self.ensure_fit(name, trial, 43, session, fold, "screen")
        self.assert_unchanged()
        rankings = {
            name: plan.rank_trials(
                self.collect(name, plan.TRIALS, plan.SCREEN_FOLDS),
                plan.TRIALS,
                SESSIONS,
                plan.SCREEN_FOLDS,
            )
            for name in plan.MODELS
        }
        promoted = {
            name: [row["trial"] for row in rankings[name][: plan.TOP_K]]
            for name in plan.MODELS
        }
        self.freeze(
            "screen_selection.json",
            {"signature": self.signature, "rankings": rankings, "promoted": promoted},
        )
        print(f"SCREEN COMPLETE: {promoted}", flush=True)

        for session in SESSIONS:
            for fold in plan.FOLDS:
                if fold in plan.SCREEN_FOLDS:
                    continue
                for name in plan.MODELS:
                    for trial in promoted[name]:
                        self.ensure_fit(name, trial, 43, session, fold, "confirm")
        self.assert_unchanged()
        rankings = {
            name: plan.rank_trials(
                self.collect(name, promoted[name], plan.FOLDS),
                promoted[name],
                SESSIONS,
                plan.FOLDS,
            )
            for name in plan.MODELS
        }
        finalists = {name: rankings[name][0]["trial"] for name in plan.MODELS}
        winner = sorted(
            plan.MODELS,
            key=lambda name: (
                -rankings[name][0]["validation_r2_mean"],
                rankings[name][0]["validation_loss_mean"],
                name,
            ),
        )[0]
        selection = {
            "signature": self.signature,
            "rankings": rankings,
            "finalists": finalists,
            "validation_selected_architecture": winner,
            "selection_seed": 43,
            "test_used_for_selection": False,
        }
        self.freeze("final_selection.json", selection)
        print(
            f"FINAL CONFIGURATIONS FROZEN: {finalists}; validation winner={winner}",
            flush=True,
        )

        for session in SESSIONS:
            for fold in plan.FOLDS:
                for seed in plan.SEEDS[1:]:
                    for name in plan.MODELS:
                        self.ensure_fit(
                            name, finalists[name], seed, session, fold, "seed_check"
                        )
        self.assert_unchanged()
        rows = [
            row
            for name in plan.MODELS
            for row in self.collect(name, [finalists[name]], plan.FOLDS, plan.SEEDS)
        ]
        plan.require_test_gate(finalists, rows, SESSIONS)
        self.freeze(
            "test_gate.json",
            {
                "signature": self.signature,
                "selection_sha256": protocol.sha256_file(
                    OUTPUT / "final_selection.json"
                ),
                "checkpoints": {
                    row["checkpoint"]: row["checkpoint_sha256"] for row in rows
                },
                "verified_finalist_checkpoints": len(rows),
                "test_selection_permitted": False,
            },
        )
        self.test_finalists(selection, rows)

    def test_finalists(self, selection, rows):
        tests = []
        for session in SESSIONS:
            for fold in plan.FOLDS:
                data, evidence = prepare(session, fold)
                for row in (
                    r for r in rows if r["session"] == session and r["fold"] == fold
                ):
                    if (OUTPUT / "STOP_AFTER_FOLD").exists():
                        raise StopRequested(
                            "STOP_AFTER_FOLD requested during final evaluation"
                        )
                    name, trial, seed = row["model"], row["trial"], row["seed"]
                    self.progress(
                        "test_finalists",
                        f"{name}/{trial}/seed{seed}/{session}/fold{fold}",
                    )
                    path = OUTPUT / row["checkpoint"]
                    saved = original.load_training_checkpoint(path)
                    check_saved(
                        saved,
                        identity(name, trial, seed, session, fold, self.signature),
                        evidence,
                    )
                    out = (
                        path.parent.parent
                        / "test_results"
                        / f"{session}_fold{fold}.json"
                    )
                    prediction_path = out.with_suffix(".npz")
                    if out.exists():
                        result = read_json(out)
                        if any(result.get(k) != v for k, v in row.items()):
                            raise ValueError("Saved test receipt changed")
                        if (
                            protocol.sha256_file(prediction_path)
                            != result["prediction_sha256"]
                        ):
                            raise ValueError("Saved test prediction hash mismatch")
                        with np.load(prediction_path, allow_pickle=False) as stored:
                            if (
                                not np.array_equal(stored["bins"], data.test_bins)
                                or not np.array_equal(
                                    stored["target"], data.velocity[data.test_bins]
                                )
                                or str(stored["checkpoint_sha256"].item())
                                != row["checkpoint_sha256"]
                            ):
                                raise ValueError(
                                    "Test prediction identity/targets changed"
                                )
                            if (
                                protocol.metrics(stored["target"], stored["prediction"])
                                != result["test"]
                            ):
                                raise ValueError(
                                    "Test metrics differ from saved predictions"
                                )
                    else:
                        model = (
                            create_model(name, plan.TRIALS[trial])
                            .to(self.device)
                            .eval()
                        )
                        model.load_state_dict(saved["model_state"])
                        prediction = protocol.predict_last(
                            model,
                            data.normalized_features,
                            data.test_bins,
                            data.target_mean,
                            data.target_std,
                            self.device,
                            plan.FIXED["batch_size"],
                        )
                        if not np.isfinite(prediction).all():
                            raise FloatingPointError("Nonfinite test predictions")
                        prediction_path.parent.mkdir(parents=True, exist_ok=True)
                        contract.session_data.save_npz_atomic(
                            prediction_path,
                            prediction=prediction,
                            target=data.velocity[data.test_bins],
                            bins=data.test_bins,
                            checkpoint_sha256=np.asarray(row["checkpoint_sha256"]),
                        )
                        result = {
                            **row,
                            "test": protocol.metrics(
                                data.velocity[data.test_bins], prediction
                            ),
                            "prediction_file": prediction_path.relative_to(
                                OUTPUT
                            ).as_posix(),
                            "prediction_sha256": protocol.sha256_file(prediction_path),
                        }
                        protocol.write_json_atomic(out, result)
                        del model
                    tests.append(result)
        self.write_report(selection, tests)

    def write_report(self, selection, tests):
        plan.require_test_gate(selection["finalists"], tests, SESSIONS)
        historical = original.historical_scores()
        # Previous test scores enter only the final report, after configuration
        # selection and all new test predictions have already been frozen.
        previous_metrics = read_json(PREVIOUS / "metrics.json")
        previous_scores = {
            (row["session"], row["fold"]): row["test"]["r2_mean"]
            for row in previous_metrics["results"]
        }
        expected_pairs = {
            (session, fold) for session in SESSIONS for fold in plan.FOLDS
        }
        if (
            previous_metrics["status"] != "complete"
            or previous_metrics.get("full_30fold") is not True
            or len(previous_metrics["results"]) != len(expected_pairs)
            or set(previous_scores) != expected_pairs
            or not all(math.isfinite(value) for value in previous_scores.values())
        ):
            raise ValueError("Prior large-model test comparison is incomplete")
        summary = []
        session_summary = []
        for name in plan.MODELS:
            for seed in plan.SEEDS:
                selected = [
                    r for r in tests if r["model"] == name and r["seed"] == seed
                ]
                values = [r["test"]["r2_mean"] for r in selected]
                deltas = [
                    r["test"]["r2_mean"] - historical[(r["session"], r["fold"])]
                    for r in selected
                ]
                previous_deltas = [
                    row["test"]["r2_mean"]
                    - previous_scores[(row["session"], row["fold"])]
                    for row in selected
                ]
                summary.append(
                    {
                        "model": name,
                        "trial": selection["finalists"][name],
                        "seed": seed,
                        "folds": len(values),
                        "test_r2_mean": statistics.mean(values),
                        "test_r2_sample_sd": statistics.stdev(values),
                        "worst_fold_r2": min(values),
                        "delta_vs_historical_midsize": statistics.mean(deltas),
                        "wins_vs_historical": sum(x > 0 for x in deltas),
                        "delta_vs_previous_large": statistics.mean(previous_deltas),
                        "wins_vs_previous_large": sum(x > 0 for x in previous_deltas),
                        "parameters": self.config["capacity"][name]["parameters"],
                        "fp32_weight_kib": self.config["capacity"][name][
                            "fp32_weight_kib"
                        ],
                    }
                )
                for session in SESSIONS:
                    scores = [
                        r["test"]["r2_mean"]
                        for r in selected
                        if r["session"] == session
                    ]
                    session_summary.append(
                        {
                            "model": name,
                            "seed": seed,
                            "session": session,
                            "test_r2_mean": statistics.mean(scores),
                            "test_r2_sample_sd": statistics.stdev(scores),
                        }
                    )
        metrics = {
            "status": "complete",
            "full_30fold": True,
            "full_30fold_each_seed": True,
            "finalist_test_folds": len(tests),
            "expected_training_fits": plan.expected_fits()["total"],
            "config": self.config,
            "selection": selection,
            "summary": summary,
            "sessions": session_summary,
            "results": tests,
            "previous_large_test_comparison_sha256": protocol.sha256_file(
                PREVIOUS / "metrics.json"
            ),
        }
        protocol.write_csv(OUTPUT / "summary.csv", summary)
        protocol.write_csv(OUTPUT / "sessions.csv", session_summary)
        text = [
            "# Phase18 fixed-capacity head regularization tuning",
            "",
            "All 96 fits and 30 finalist test-fold artifacts verified.",
            "",
            f"Validation-selected architecture (seed 43, before test): **{selection['validation_selected_architecture']}**.",
            "",
            "| Model | Trial | Seed | Test R2 mean +/- sample SD | Delta vs historical Midsize | Parameters | FP32 KiB |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in summary:
            text.append(
                f"| {row['model']} | {row['trial']} | {row['seed']} | {row['test_r2_mean']:.4f} +/- {row['test_r2_sample_sd']:.4f} | {row['delta_vs_historical_midsize']:+.4f} | {row['parameters']:,} | {row['fp32_weight_kib']:.2f} |"
            )
            text.append(
                f"\nPaired test R2 change versus previous large minGRU: {row['delta_vs_previous_large']:+.4f}; wins {row['wins_vs_previous_large']}/30.\n"
            )
        text += [
            "",
            "Hyperparameters were frozen before test. Seed 43 only; no additional-seed robustness claim. Checkpoints are selected by validation normalized MSE, configurations by validation R2.",
            "",
            self.config["limitations"],
            "",
            "Stage-one selection uses one prespecified fold per session and can miss configurations that would win a full search. FP32 GPU execution does not establish STM32 latency or physical SDRAM transfer performance.",
        ]
        (OUTPUT / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
        protocol.write_json_atomic(OUTPUT / "metrics.json", metrics)
        self.progress("complete", status="complete", finalist_test_folds=len(tests))
        print(f"COMPLETE: {OUTPUT / 'REPORT.md'}", flush=True)


def preflight(config, device):
    protocol.seed_everything(43, mps=False)
    for trial_id in ("r00", "r03"):
        model = create_model(plan.MODELS[0], plan.TRIALS[trial_id]).to(device).eval()
        inputs = torch.randn(2, 192, 50, device=device)
        with torch.no_grad():
            actual = model(inputs)
            reference = model.forward_reference(inputs)[:, -1:]
            sequential = model.forward_sequential(inputs)
            if actual.shape != (2, 1, 2) or not torch.isfinite(actual).all():
                raise ValueError("Last-only model output failed preflight")
            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(actual, sequential, rtol=1e-4, atol=1e-5)
        model.train()
        model(inputs)[:, -1].square().mean().backward()
        if any(
            p.grad is None or not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise ValueError("Large-model gradient preflight failed")
        del model
    folds = {}
    for session in SESSIONS:
        for fold in plan.FOLDS:
            _, evidence = prepare(session, fold)
            folds[f"{session}|{fold}"] = evidence
            print(f"PREFLIGHT verified {session} fold {fold}", flush=True)
    protocol.write_json_atomic(
        OUTPUT / "preflight.json",
        {
            "signature": digest(config),
            "folds": folds,
            "verified_folds": len(folds),
            "optimizer_steps": 0,
            "last_only_and_sequential_equivalence": True,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    contract.verify_protocol_lock()
    contract.configure_paths(
        contract.INDY / "data", contract.default_gui_root(), OUTPUT / ".cache"
    )
    device = protocol.select_device(args.device)
    config = build_config(args.device, args.threads)
    signature = digest(config)
    # Shared sweep lock plus locks respected by the original three entry points.
    with ExitStack() as stack:
        stack.enter_context(original.exclusive_run(RESULTS / ".phase17_gpu"))
        for name in ("mingru", "mamba2", "transformer"):
            stack.enter_context(original.exclusive_run(RESULTS / name))
        stack.enter_context(original.exclusive_run(PREVIOUS))
        stack.enter_context(original.exclusive_run(OUTPUT))
        path = OUTPUT / "config.json"
        if path.exists():
            if digest(read_json(path)) != signature:
                raise ValueError(
                    "Cannot resume: configuration/code/environment/input changed"
                )
            if not args.resume and not args.preflight_only:
                raise FileExistsError("Existing sweep: use --resume")
        elif args.resume:
            raise FileNotFoundError("No saved sweep to resume")
        else:
            protocol.write_json_atomic(path, config)
        sweep = Sweep(config, device)
        try:
            receipt = OUTPUT / "preflight.json"
            if args.preflight_only or not receipt.exists():
                sweep.progress("preflight")
                preflight(config, device)
            elif (
                read_json(receipt)["signature"] != signature
                or read_json(receipt)["verified_folds"] != 30
            ):
                raise ValueError("Invalid preflight receipt")
            if args.preflight_only:
                sweep.progress("ready", status="ready")
                print("PREFLIGHT PASSED: 30 folds; zero optimizer steps", flush=True)
                return
            sweep.run()
        except StopRequested as error:
            sweep.progress("stopped", status="stopped", reason=str(error))
            print(str(error), flush=True)
        except BaseException as error:
            sweep.progress(
                "failed",
                status="failed",
                error=str(error),
                traceback=traceback.format_exc(),
            )
            raise


if __name__ == "__main__":
    main()
