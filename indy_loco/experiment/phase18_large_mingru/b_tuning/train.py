"""Tune fixed minGRU B with EMA-only selection and an immutable cached control."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import math
import os
import statistics
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from indy_loco.experiment.phase18_large_mingru.ab_study import train as ab
from indy_loco.experiment.phase18_large_mingru.b_tuning import plan

protocol, contract, original, models = ab.protocol, ab.contract, ab.original, ab.models
read_json, digest, prepare = ab.read_json, ab.digest, ab.prepare
fit, select_policy = ab.fit, ab.select_policy
SESSIONS, RESULTS = ab.SESSIONS, ab.RESULTS
OUTPUT = PHASE / "results/b_tuning_v1"
BASELINE_OUTPUT = ab.OUTPUT


def create_model(trial):
    return ab.create_model(plan.MODEL, plan.TRIALS[trial])


def checkpoint_path(trial, seed, session, fold):
    return (
        OUTPUT
        / "runs"
        / plan.MODEL
        / trial
        / f"seed{seed}"
        / "checkpoints"
        / f"{session}_fold{fold}.pt"
    )


def identity(trial, seed, session, fold, signature):
    return ab.identity(plan.MODEL, trial, seed, session, fold, signature)


def check_saved(saved, expected, evidence):
    ab.check_saved(saved, expected, evidence)
    if saved.get("hyperparameters") != plan.TRIALS[expected["trial"]]:
        raise ValueError("Saved hyperparameters differ from frozen trial")
    if saved.get("training") != plan.FIXED:
        raise ValueError("Saved training schedule changed")


def training_receipt(saved, path):
    """All new receipts use repository-relative paths, including cached B0."""
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
        "policy_validation": {
            policy: {
                key: value for key, value in candidate.items() if key != "model_state"
            }
            for policy, candidate in saved["weight_policies"].items()
        },
        "checkpoint": path.relative_to(REPO).as_posix(),
        "checkpoint_sha256": protocol.sha256_file(path),
    }


def verify_baseline_fold(session, fold, source_config):
    """Read only cached training/validation artifacts; leave prior test closed."""
    path = ab.checkpoint_path(plan.MODEL, "t00", 43, session, fold)
    saved = original.load_training_checkpoint(path)
    _, evidence = prepare(session, fold)
    ab.check_saved(
        saved,
        ab.identity(plan.MODEL, "t00", 43, session, fold, digest(source_config)),
        evidence,
    )
    if (
        saved.get("hyperparameters") != plan.TRIALS[plan.BASELINE]
        or saved.get("training") != plan.FIXED
    ):
        raise ValueError("Cached B recipe no longer equals the prescribed control")
    receipt = ab.training_receipt(saved, path)
    if read_json(path.with_suffix(".validation.json")) != receipt:
        raise ValueError("Cached B validation receipt differs from verified checkpoint")
    return saved, receipt


def baseline_provenance():
    source_config = read_json(BASELINE_OUTPUT / "config.json")
    selection = read_json(BASELINE_OUTPUT / "final_selection.json")
    if (
        selection["signature"] != digest(source_config)
        or selection["finalists"].get(plan.MODEL) != "t00"
        or selection["weight_policies"].get(plan.MODEL) != plan.WEIGHT_POLICY
        or selection.get("test_used_for_selection") is not False
    ):
        raise ValueError("Cached baseline selection identity changed")
    if (
        source_config["fixed"] != plan.FIXED
        or source_config["trials"]["t00"] != plan.TRIALS[plan.BASELINE]
    ):
        raise ValueError("Tuning is not an exact extension of cached B's recipe")
    for relative, expected in source_config["code_sha256"].items():
        if (
            hashlib.sha256(
                (REPO / relative).read_text(encoding="utf-8").encode()
            ).hexdigest()
            != expected
        ):
            raise ValueError(f"Frozen baseline code changed: {relative}")
    hashes = {}
    for session in SESSIONS:
        for fold in plan.FOLDS:
            _, receipt = verify_baseline_fold(session, fold, source_config)
            path = BASELINE_OUTPUT / receipt["checkpoint"]
            hashes[path.relative_to(REPO).as_posix()] = receipt["checkpoint_sha256"]
            hashes[
                path.with_suffix(".validation.json").relative_to(REPO).as_posix()
            ] = protocol.sha256_file(path.with_suffix(".validation.json"))
    return {
        "config_sha256": protocol.sha256_file(BASELINE_OUTPUT / "config.json"),
        "selection_sha256": protocol.sha256_file(
            BASELINE_OUTPUT / "final_selection.json"
        ),
        "source_signature": digest(source_config),
        "verified_validation_folds": 30,
        "training_artifact_sha256": hashes,
        "test_scores_read_for_selection": False,
    }, source_config


def build_config(device, threads):
    if plan.FIXED != ab.plan.FIXED:
        raise ValueError("Reused fit implementation and frozen training plan differ")
    provenance, prior = baseline_provenance()
    args = SimpleNamespace(
        model="mingru",
        device=device,
        threads=threads,
        indy_cache_root=ab.CACHE_ROOT,
        gui_root=contract.default_gui_root(),
    )
    base = original.make_config(
        args, list(SESSIONS), list(plan.FOLDS), include_data=True
    )
    if base["inputs"] != prior["inputs"] or base["references"] != prior["references"]:
        raise ValueError("Input/reference fingerprints differ from published B")
    code = dict(prior["code_sha256"])
    for path in sorted(HERE.glob("*.py")):
        code[path.relative_to(REPO).as_posix()] = hashlib.sha256(
            path.read_text(encoding="utf-8").encode()
        ).hexdigest()
    return {
        "phase": "phase18_large_mingru",
        "extension": "b_tuning_v1",
        "model": plan.MODEL,
        "trials": plan.TRIALS,
        "new_trials": list(plan.NEW_TRIALS),
        "baseline": plan.BASELINE,
        "fixed": plan.FIXED,
        "weight_policy": plan.WEIGHT_POLICY,
        "sessions": list(SESSIONS),
        "folds": list(plan.FOLDS),
        "seeds": list(plan.SEEDS),
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
        "baseline_provenance": provenance,
        "code_sha256": code,
        "capacity": {
            key: value
            for key, value in models.capacity_report(plan.MODEL).items()
            if key != "parameter_groups"
        },
        "selection": "EMA only; 30 new screening fits, top two new recipes complete all 30 folds (48 additional fits). Full-validation ranking includes immutable B0, preferring B0 on equal R2. Freeze winner before test; evaluate both new finalists and verify cached B0 predictions only after all 78 new fits. No additional seeds.",
        "limitations": "Single seed 43, six sessions and correlated fivefold splits; repeated global HPO is not nested unbiased estimation. Previously exposed test values never enter selection. GPU FP32 does not establish STM32 latency or SDRAM placement. Architecture fixed at 374402 parameters.",
    }


def learning_curve_diagnostics(rows):
    """Summarize frozen EMA histories without feeding diagnostics into selection."""
    folds = []
    for row in rows:
        path = REPO / row["checkpoint"]
        if protocol.sha256_file(path) != row["checkpoint_sha256"]:
            raise ValueError("Learning curve checkpoint changed")
        history = original.load_training_checkpoint(path)["history"]
        best = history[row["best_epoch"] - 1]
        last = history[-1]
        best_loss = best["policy_validation"][plan.WEIGHT_POLICY]["normalized_loss"]
        if best_loss != row["validation"]["normalized_loss"]:
            raise ValueError("EMA best epoch differs from checkpoint receipt")
        best_probe = best["policy_training_eval_probe"][plan.WEIGHT_POLICY][
            "normalized_loss"
        ]
        last_probe = last["policy_training_eval_probe"][plan.WEIGHT_POLICY][
            "normalized_loss"
        ]
        last_loss = last["policy_validation"][plan.WEIGHT_POLICY]["normalized_loss"]
        folds.append(
            {
                "trial": row["trial"],
                "session": row["session"],
                "fold": row["fold"],
                "ema_best_epoch": row["best_epoch"],
                "stop_epoch": len(history),
                "reached_epoch_cap": len(history) == plan.FIXED["epochs"],
                "best_validation_loss": best_loss,
                "last_validation_loss": last_loss,
                "validation_loss_best_to_last_change": last_loss - best_loss,
                "best_training_probe_loss": best_probe,
                "last_training_probe_loss": last_probe,
                "generalization_gap_at_best": best_loss - best_probe,
                "generalization_gap_at_last": last_loss - last_probe,
            }
        )
    summaries = []
    for trial in dict.fromkeys(row["trial"] for row in folds):
        selected = [row for row in folds if row["trial"] == trial]
        fields = (
            "best_validation_loss",
            "last_validation_loss",
            "validation_loss_best_to_last_change",
            "best_training_probe_loss",
            "last_training_probe_loss",
            "generalization_gap_at_best",
            "generalization_gap_at_last",
        )
        summaries.append(
            {
                "trial": trial,
                "folds": len(selected),
                "ema_best_epoch_median": statistics.median(
                    row["ema_best_epoch"] for row in selected
                ),
                "stop_epoch_median": statistics.median(
                    row["stop_epoch"] for row in selected
                ),
                "epoch_cap_count": sum(row["reached_epoch_cap"] for row in selected),
                "epoch_cap_fraction": statistics.mean(
                    row["reached_epoch_cap"] for row in selected
                ),
                **{
                    f"{field}_mean": statistics.mean(row[field] for row in selected)
                    for field in fields
                },
            }
        )
    return {
        "summary": summaries,
        "folds": folds,
        "used_for_selection": False,
        "caveat": "Best-to-last validation deterioration is partly induced by checkpoint selection and early stopping. Probe uses at most 1024 fixed dropout-off training bins; it is diagnostic, not independent evidence of causation.",
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
                "verified_new_fits_this_invocation": len(self.verified),
                "completed_new_fit_receipts": len(
                    list(
                        (OUTPUT / "runs").glob(
                            "*/b*/seed*/checkpoints/*.validation.json"
                        )
                    )
                ),
                "cached_baseline_folds": 30,
                "planned_new_fits": plan.expected_fits(),
                "updated_at_utc": protocol.utc_now(),
                "pid": os.getpid(),
                **extra,
            },
        )

    def stop_check(self):
        if (OUTPUT / "STOP_AFTER_FOLD").exists():
            raise StopRequested(
                "STOP_AFTER_FOLD requested; completed checkpoints retained"
            )

    def assert_unchanged(self):
        if (
            digest(build_config(self.config["device"], self.config["threads"]))
            != self.signature
        ):
            raise ValueError(
                "Frozen code, environment, input, baseline or reference changed"
            )

    def freeze(self, filename, payload):
        path = OUTPUT / filename
        if path.exists() and read_json(path) != payload:
            raise ValueError(f"Frozen artifact changed: {path}")
        protocol.write_json_atomic(path, payload)

    def ensure_fit(self, trial, seed, session, fold, stage):
        if trial not in plan.NEW_TRIALS:
            raise ValueError("Cached baseline must never be retrained or overwritten")
        self.stop_check()
        label = f"{plan.MODEL}/{trial}/seed{seed}/{session}/fold{fold}"
        self.progress(stage, label)
        data, evidence = prepare(session, fold)
        expected = identity(trial, seed, session, fold, self.signature)
        path = checkpoint_path(trial, seed, session, fold)
        if path.exists():
            saved = original.load_training_checkpoint(path)
            check_saved(saved, expected, evidence)
            print(f"RESUME verified {label}", flush=True)
        else:
            protocol.seed_everything(seed, mps=False)
            model = create_model(trial).to(self.device)
            # Preserve both original initialization and the post-construction RNG reset.
            protocol.seed_everything(seed, mps=False)
            fitted = fit(
                model,
                plan.MODEL,
                plan.TRIALS[trial],
                seed,
                data,
                self.device,
                path.parent.parent / "epochs" / f"{session}_fold{fold}.json",
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
            check_saved(saved, expected, evidence)
            original.save_training_checkpoint(path, saved)
            del model
        receipt = training_receipt(saved, path)
        out = path.with_suffix(".validation.json")
        if out.exists() and read_json(out) != receipt:
            raise ValueError("Saved validation receipt was modified")
        protocol.write_json_atomic(out, receipt)
        self.verified.add(label)
        self.progress(stage, label)
        return select_policy(receipt, plan.WEIGHT_POLICY)

    def baseline_row(self, session, fold):
        source = read_json(BASELINE_OUTPUT / "config.json")
        _, receipt = verify_baseline_fold(session, fold, source)
        path = BASELINE_OUTPUT / receipt["checkpoint"]
        selected = select_policy(receipt, plan.WEIGHT_POLICY)
        return {
            **selected,
            "trial": plan.BASELINE,
            "signature": self.signature,
            "source_trial": "t00",
            "source_signature": receipt["signature"],
            "cached_control": True,
            "checkpoint": path.relative_to(REPO).as_posix(),
        }

    def collect(self, trials, folds, seeds=(43,)):
        rows = []
        for trial in trials:
            for seed in seeds:
                for session in SESSIONS:
                    for fold in folds:
                        if trial == plan.BASELINE:
                            if seed != 43:
                                raise ValueError(
                                    "No cached baseline for additional seed"
                                )
                            rows.append(self.baseline_row(session, fold))
                            continue
                        path = checkpoint_path(trial, seed, session, fold)
                        receipt = read_json(path.with_suffix(".validation.json"))
                        saved = original.load_training_checkpoint(path)
                        _, evidence = prepare(session, fold)
                        check_saved(
                            saved,
                            identity(trial, seed, session, fold, self.signature),
                            evidence,
                        )
                        if training_receipt(saved, path) != receipt:
                            raise ValueError(
                                "Validation receipt differs from checkpoint"
                            )
                        rows.append(select_policy(receipt, plan.WEIGHT_POLICY))
        return rows

    def run(self):
        baseline = self.collect([plan.BASELINE], plan.FOLDS)
        self.freeze(
            "baseline_validation.json",
            {"signature": self.signature, "results": baseline},
        )
        for session in SESSIONS:
            for fold in plan.SCREEN_FOLDS:
                for trial in plan.NEW_TRIALS:
                    self.ensure_fit(trial, 43, session, fold, "screen")
        self.stop_check()
        self.assert_unchanged()
        screening = self.collect(plan.NEW_TRIALS, plan.SCREEN_FOLDS)
        control_screen = [row for row in baseline if row["fold"] in plan.SCREEN_FOLDS]
        rankings = plan.rank_trials(
            screening + control_screen, plan.TRIALS, SESSIONS, plan.SCREEN_FOLDS
        )
        promoted = plan.promote_candidates(screening, SESSIONS)
        self.freeze(
            "screen_selection.json",
            {
                "signature": self.signature,
                "rankings": rankings,
                "promoted": promoted,
                "weight_policy": plan.WEIGHT_POLICY,
                "test_used_for_selection": False,
            },
        )
        self.write_validation_report("screen", screening + control_screen, rankings)
        print(
            f"SCREEN COMPLETE: promoted={promoted}; EMA policy remains frozen",
            flush=True,
        )
        for session in SESSIONS:
            for fold in plan.FOLDS:
                if fold not in plan.SCREEN_FOLDS:
                    for trial in promoted:
                        self.ensure_fit(trial, 43, session, fold, "confirm")
        self.stop_check()
        self.assert_unchanged()
        finalists = [plan.BASELINE, *promoted]
        rows = self.collect(finalists, plan.FOLDS)
        rankings = plan.rank_trials(rows, finalists, SESSIONS, plan.FOLDS)
        selection = {
            "signature": self.signature,
            "rankings": rankings,
            "finalists": finalists,
            "validation_selected_trial": rankings[0]["trial"],
            "weight_policy": plan.WEIGHT_POLICY,
            "selection_seed": 43,
            "test_used_for_selection": False,
        }
        self.freeze("final_selection.json", selection)
        self.write_validation_report("full_validation", rows, rankings)
        # Explicitly verify every one of the 78 unique new fits before opening test.
        all_new = self.collect(plan.NEW_TRIALS, plan.SCREEN_FOLDS)
        all_new += self.collect(
            promoted,
            tuple(fold for fold in plan.FOLDS if fold not in plan.SCREEN_FOLDS),
        )
        if len(all_new) != plan.expected_fits()["total"] or len(
            {row["checkpoint"] for row in all_new}
        ) != len(all_new):
            raise ValueError("Test remains closed: all 78 unique new fits required")
        plan.require_test_gate(finalists, rows, SESSIONS)
        self.freeze(
            "test_gate.json",
            {
                "signature": self.signature,
                "selection_sha256": protocol.sha256_file(
                    OUTPUT / "final_selection.json"
                ),
                "verified_new_training_fits": len(all_new),
                "verified_finalist_checkpoints": len(rows),
                "new_training_checkpoints": {
                    row["checkpoint"]: row["checkpoint_sha256"] for row in all_new
                },
                "finalist_checkpoints": {
                    row["checkpoint"]: row["checkpoint_sha256"] for row in rows
                },
                "weight_policy": plan.WEIGHT_POLICY,
                "test_selection_permitted": False,
            },
        )
        print(
            f"FINAL SELECTION FROZEN: {selection['validation_selected_trial']}; opening final test",
            flush=True,
        )
        self.test_finalists(selection, rows)

    def write_validation_report(self, stage, rows, ranking):
        controls = {
            (row["session"], row["fold"]): row
            for row in rows
            if row["trial"] == plan.BASELINE
        }
        paired = []
        for rank in ranking:
            selected = [row for row in rows if row["trial"] == rank["trial"]]
            deltas = [
                row["validation"]["r2_mean"]
                - controls[(row["session"], row["fold"])]["validation"]["r2_mean"]
                for row in selected
            ]
            paired.append(
                {
                    **rank,
                    "paired_delta_vs_b0": statistics.mean(deltas),
                    "wins_vs_b0": sum(delta > 0 for delta in deltas),
                    "sessions": {
                        session: statistics.mean(
                            row["validation"]["r2_mean"]
                            for row in selected
                            if row["session"] == session
                        )
                        for session in SESSIONS
                    },
                    "axis_r2": {
                        axis: statistics.mean(
                            row["validation"][axis] for row in selected
                        )
                        for axis in ("r2_x", "r2_y")
                    },
                }
            )
        protocol.write_json_atomic(
            OUTPUT / f"{stage}_validation.json",
            {
                "signature": self.signature,
                "summary": paired,
                "learning_curves": learning_curve_diagnostics(rows),
                "results": rows,
                "test_used": False,
            },
        )

    @staticmethod
    def verify_prediction(path, result, row, data):
        if protocol.sha256_file(path) != result["prediction_sha256"]:
            raise ValueError("Test prediction hash mismatch")
        with np.load(path, allow_pickle=False) as stored:
            if (
                not np.array_equal(stored["bins"], data.test_bins)
                or not np.array_equal(stored["target"], data.velocity[data.test_bins])
                or str(stored["checkpoint_sha256"].item()) != row["checkpoint_sha256"]
                or str(stored["weight_policy"].item()) != plan.WEIGHT_POLICY
            ):
                raise ValueError("Test prediction identity, targets or policy changed")
            if (
                not np.isfinite(stored["prediction"]).all()
                or protocol.metrics(stored["target"], stored["prediction"])
                != result["test"]
            ):
                raise ValueError("Saved test scores differ from finite predictions")

    def test_finalists(self, selection, rows):
        plan.require_test_gate(selection["finalists"], rows, SESSIONS)
        gate = read_json(OUTPUT / "test_gate.json")
        if (
            gate["signature"] != self.signature
            or gate["verified_new_training_fits"] != plan.expected_fits()["total"]
            or gate["selection_sha256"]
            != protocol.sha256_file(OUTPUT / "final_selection.json")
        ):
            raise ValueError("Final test gate was not frozen correctly")
        tests = []
        for session in SESSIONS:
            for fold in plan.FOLDS:
                data, evidence = prepare(session, fold)
                for row in (
                    item
                    for item in rows
                    if item["session"] == session and item["fold"] == fold
                ):
                    self.stop_check()
                    trial, seed = row["trial"], row["seed"]
                    self.progress(
                        "test_finalists",
                        f"{trial}/{session}/fold{fold}",
                        completed_test_folds=len(tests),
                    )
                    path = REPO / row["checkpoint"]
                    if protocol.sha256_file(path) != row["checkpoint_sha256"]:
                        raise ValueError("Finalist checkpoint changed before test")
                    if trial == plan.BASELINE:
                        if self.baseline_row(session, fold) != row:
                            raise ValueError("Cached B0 validation identity changed")
                        source_out = (
                            path.parent.parent
                            / "test_results"
                            / f"{session}_fold{fold}.json"
                        )
                        source_result = read_json(source_out)
                        source_receipt = select_policy(
                            read_json(path.with_suffix(".validation.json")),
                            plan.WEIGHT_POLICY,
                        )
                        if any(
                            source_result.get(key) != value
                            for key, value in source_receipt.items()
                        ):
                            raise ValueError("Cached B0 test receipt identity mismatch")
                        prediction_path = (
                            BASELINE_OUTPUT / source_result["prediction_file"]
                        )
                        self.verify_prediction(
                            prediction_path, source_result, row, data
                        )
                        result = {
                            **row,
                            "test": source_result["test"],
                            "prediction_file": prediction_path.relative_to(
                                REPO
                            ).as_posix(),
                            "prediction_sha256": source_result["prediction_sha256"],
                            "source_test_receipt_sha256": protocol.sha256_file(
                                source_out
                            ),
                        }
                        self.freeze(f"baseline_test/{session}_fold{fold}.json", result)
                    else:
                        saved = original.load_training_checkpoint(path)
                        check_saved(
                            saved,
                            identity(trial, seed, session, fold, self.signature),
                            evidence,
                        )
                        if (
                            select_policy(
                                training_receipt(saved, path), plan.WEIGHT_POLICY
                            )
                            != row
                        ):
                            raise ValueError("Selected EMA receipt changed before test")
                        out = (
                            path.parent.parent
                            / "test_results"
                            / f"{session}_fold{fold}.json"
                        )
                        prediction_path = out.with_suffix(".npz")
                        if out.exists():
                            result = read_json(out)
                            if any(
                                result.get(key) != value for key, value in row.items()
                            ):
                                raise ValueError("Saved test receipt changed")
                            self.verify_prediction(prediction_path, result, row, data)
                        else:
                            model = create_model(trial).to(self.device).eval()
                            model.load_state_dict(
                                saved["weight_policies"][plan.WEIGHT_POLICY][
                                    "model_state"
                                ]
                            )
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
                                raise FloatingPointError("Nonfinite test prediction")
                            prediction_path.parent.mkdir(parents=True, exist_ok=True)
                            contract.session_data.save_npz_atomic(
                                prediction_path,
                                prediction=prediction,
                                target=data.velocity[data.test_bins],
                                bins=data.test_bins,
                                checkpoint_sha256=np.asarray(row["checkpoint_sha256"]),
                                weight_policy=np.asarray(plan.WEIGHT_POLICY),
                            )
                            result = {
                                **row,
                                "test": protocol.metrics(
                                    data.velocity[data.test_bins], prediction
                                ),
                                "prediction_file": prediction_path.relative_to(
                                    REPO
                                ).as_posix(),
                                "prediction_sha256": protocol.sha256_file(
                                    prediction_path
                                ),
                            }
                            protocol.write_json_atomic(out, result)
                            del model
                    tests.append(result)
        self.write_report(selection, tests)

    def write_report(self, selection, tests):
        plan.require_test_gate(selection["finalists"], tests, SESSIONS)
        controls = {
            (row["session"], row["fold"]): row["test"]["r2_mean"]
            for row in tests
            if row["trial"] == plan.BASELINE
        }
        # Historical test scores enter only after HPO has frozen and new tests are complete.
        historical = original.historical_scores()
        previous_metrics = read_json(ab.PREVIOUS / "metrics.json")
        previous = {
            (row["session"], row["fold"]): row["test"]["r2_mean"]
            for row in previous_metrics["results"]
        }
        expected_pairs = {
            (session, fold) for session in SESSIONS for fold in plan.FOLDS
        }
        if (
            previous_metrics.get("status") != "complete"
            or previous_metrics.get("full_30fold") is not True
            or len(previous_metrics["results"]) != 30
            or set(previous) != expected_pairs
            or not expected_pairs.issubset(historical)
            or not all(math.isfinite(value) for value in previous.values())
        ):
            raise ValueError("Historical paired test comparison incomplete")
        summary, sessions = [], []
        for trial in selection["finalists"]:
            selected = [row for row in tests if row["trial"] == trial]
            values = [row["test"]["r2_mean"] for row in selected]
            deltas = [
                row["test"]["r2_mean"] - controls[(row["session"], row["fold"])]
                for row in selected
            ]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Nonfinite final test scores")
            rank = next(
                item for item in selection["rankings"] if item["trial"] == trial
            )
            summary.append(
                {
                    "trial": trial,
                    "weight_policy": plan.WEIGHT_POLICY,
                    "folds": len(values),
                    "validation_r2_mean": rank["validation_r2_mean"],
                    "test_r2_mean": statistics.mean(values),
                    "test_r2_sample_sd": statistics.stdev(values),
                    "delta_vs_b0": statistics.mean(deltas),
                    "paired_delta_sample_sd": statistics.stdev(deltas),
                    "wins_vs_b0": sum(delta > 0 for delta in deltas),
                    "delta_vs_r07": statistics.mean(
                        row["test"]["r2_mean"] - previous[(row["session"], row["fold"])]
                        for row in selected
                    ),
                    "delta_vs_historical_midsize": statistics.mean(
                        row["test"]["r2_mean"]
                        - historical[(row["session"], row["fold"])]
                        for row in selected
                    ),
                    "parameters": self.config["capacity"]["parameters"],
                    "fp32_weight_bytes": self.config["capacity"]["fp32_weight_bytes"],
                    "fp32_weight_kib": self.config["capacity"]["fp32_weight_kib"],
                }
            )
            for session in SESSIONS:
                selected_session = [
                    row for row in selected if row["session"] == session
                ]
                sessions.append(
                    {
                        "trial": trial,
                        "session": session,
                        "test_r2_mean": statistics.mean(
                            row["test"]["r2_mean"] for row in selected_session
                        ),
                        "delta_vs_b0": statistics.mean(
                            row["test"]["r2_mean"] - controls[(session, row["fold"])]
                            for row in selected_session
                        ),
                    }
                )
        curves = learning_curve_diagnostics(tests)
        metrics = {
            "status": "complete",
            "full_30fold": True,
            "finalist_test_folds": len(tests),
            "new_test_evaluations": 60,
            "verified_cached_baseline_test_folds": 30,
            "expected_new_training_fits": 78,
            "config": self.config,
            "selection": selection,
            "summary": summary,
            "sessions": sessions,
            "results": tests,
            "learning_curves": curves,
            "r07_metrics_sha256": protocol.sha256_file(ab.PREVIOUS / "metrics.json"),
        }
        protocol.write_csv(OUTPUT / "summary.csv", summary)
        protocol.write_csv(OUTPUT / "sessions.csv", sessions)
        lines = [
            "# Fixed-architecture B tuning / 固定 B 架构调参",
            "",
            "78 new fits verified; two new finalists tested on 30 folds each, plus 30 verified cached B0 test folds.",
            "",
            f"Validation-selected recipe / 验证集选择配置: **{selection['validation_selected_trial']}**. Test never changes this selection.",
            "",
            "| Trial | Validation R² | Test R² mean ± sample SD | Δ test vs B0 | Wins / 30 |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in summary:
            lines.append(
                f"| {row['trial']} | {row['validation_r2_mean']:.6f} | {row['test_r2_mean']:.6f} ± {row['test_r2_sample_sd']:.6f} | {row['delta_vs_b0']:+.6f} | {row['wins_vs_b0']} |"
            )
        for row in summary:
            lines.append(
                f"\n{row['trial']}: Δ test R² vs r07 {row['delta_vs_r07']:+.6f}; vs historical warm-started Midsize {row['delta_vs_historical_midsize']:+.6f}; FP32 weights {row['fp32_weight_bytes']:,} bytes.\n"
            )
        lines += [
            "",
            "| Trial | Median EMA best epoch | Median stop epoch | Reached 60 epochs | Best→last validation MSE change | Train/validation gap at best→last |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in curves["summary"]:
            lines.append(
                f"| {row['trial']} | {row['ema_best_epoch_median']:.1f} | {row['stop_epoch_median']:.1f} | {row['epoch_cap_count']}/{row['folds']} | {row['validation_loss_best_to_last_change_mean']:+.6f} | {row['generalization_gap_at_best_mean']:.6f} → {row['generalization_gap_at_last_mean']:.6f} |"
            )
        lines += ["", curves["caveat"]]
        lines += [
            "",
            self.config["limitations"],
            "",
            "原 B0、所有旧模型和备份保持不变。Original B0 and all previous runs/backups remain unchanged.",
            "",
        ]
        (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        protocol.write_json_atomic(OUTPUT / "metrics.json", metrics)
        self.progress("complete", status="complete", finalist_test_folds=len(tests))
        print(f"COMPLETE: {OUTPUT / 'REPORT.md'}", flush=True)


def preflight(config, device):
    protocol.seed_everything(43, mps=False)
    model = create_model(plan.BASELINE).to(device).eval()
    inputs = torch.randn(2, 192, 50, device=device)
    with torch.no_grad():
        actual = model(inputs)
        if actual.shape != (2, 1, 2) or not torch.isfinite(actual).all():
            raise ValueError("Model output preflight failed")
        torch.testing.assert_close(
            actual, model.forward_reference(inputs)[:, -1:], rtol=1e-4, atol=1e-5
        )
        torch.testing.assert_close(
            actual, model.forward_sequential(inputs), rtol=1e-4, atol=1e-5
        )
    model.train()
    model(inputs)[:, -1].square().mean().backward()
    if any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise ValueError("Model gradient preflight failed")
    del model
    folds = {}
    source = read_json(BASELINE_OUTPUT / "config.json")
    for session in SESSIONS:
        for fold in plan.FOLDS:
            saved, _ = verify_baseline_fold(session, fold, source)
            folds[f"{session}|{fold}"] = saved["preprocessing_evidence"]
            print(
                f"PREFLIGHT verified baseline and split: {session} fold {fold}",
                flush=True,
            )
    protocol.write_json_atomic(
        OUTPUT / "preflight.json",
        {
            "signature": digest(config),
            "folds": folds,
            "verified_folds": len(folds),
            "optimizer_steps": 0,
            "last_only_and_sequential_equivalence": True,
            "baseline_untouched": True,
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
    with ExitStack() as stack:
        for directory in (
            RESULTS / ".phase17_gpu",
            *(RESULTS / name for name in ("mingru", "mamba2", "transformer")),
            PHASE / "results/large_mingru_v1",
            ab.PREVIOUS,
            BASELINE_OUTPUT,
            OUTPUT,
        ):
            stack.enter_context(original.exclusive_run(directory))
        config = build_config(args.device, args.threads)
        signature = digest(config)
        path = OUTPUT / "config.json"
        if path.exists():
            if digest(read_json(path)) != signature:
                raise ValueError(
                    "Cannot resume: configuration/code/environment/input changed"
                )
            if not args.resume and not args.preflight_only:
                raise FileExistsError("Existing tuning run: use --resume")
        elif args.resume:
            raise FileNotFoundError("No saved tuning run to resume")
        else:
            protocol.write_json_atomic(path, config)
        sweep = Sweep(config, device)
        try:
            receipt = OUTPUT / "preflight.json"
            sweep.stop_check()
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
