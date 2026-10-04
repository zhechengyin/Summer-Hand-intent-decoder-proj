"""User-authorized b2-only audit/test, then quantization iff it beats original B.

This is a separate follow-up protocol, not completion of the original b2/b4
finalist study. It does not alter that study's gate or claim all78 fits finished.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import statistics
import traceback
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from indy_loco.experiment.phase18_large_mingru.b_tuning import train as tuning
from indy_loco.models.mingru_b.model import ModelB
from . import run as quant

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results/b2_completion_v1"
PACKAGE = OUTPUT / "package"
protocol, contract = tuning.protocol, tuning.contract


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    quant.write(path, value)


def audit_and_test():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for the authorized completion/test stage")
    device = torch.device("cuda")
    contract.verify_protocol_lock()
    contract.configure_paths(contract.INDY / "data", contract.default_gui_root(), OUTPUT / ".cache")
    config = tuning.read_json(tuning.OUTPUT / "config.json")
    # Verify frozen code without modifying it or weakening saved identities.
    for relative, expected in config["code_sha256"].items():
        actual = hashlib.sha256((tuning.REPO / relative).read_text(encoding="utf-8").encode()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen tuning code changed: {relative}")
    signature = tuning.digest(config)
    audited = []
    missing = [(s, f) for s in tuning.SESSIONS for f in tuning.plan.FOLDS
               if not tuning.checkpoint_path("b2", 43, s, f).is_file()]
    if missing:
        # Reuse the exact old fitting code/config and require environment identity.
        if tuning.digest(tuning.build_config(config["device"], config["threads"])) != signature:
            raise ValueError("Original training environment/config changed; cannot resume missing folds")
        class B2OnlySweep(tuning.Sweep):
            def progress(self, stage, current=None, **extra):
                write(OUTPUT / "status.json", {"status": "training_missing_b2", "stage": stage,
                      "current": current, **extra})
        sweep = B2OnlySweep(config, device)
        for session, fold in missing:
            sweep.ensure_fit("b2", 43, session, fold, "user_requested_b2_completion")
    for session in tuning.SESSIONS:
        for fold in tuning.plan.FOLDS:
            path = tuning.checkpoint_path("b2", 43, session, fold)
            saved = tuning.original.load_training_checkpoint(path)
            data, evidence = tuning.prepare(session, fold)
            tuning.check_saved(saved, tuning.identity("b2", 43, session, fold, signature), evidence)
            receipt = tuning.training_receipt(saved, path)
            if tuning.read_json(path.with_suffix(".validation.json")) != receipt:
                raise ValueError("b2 validation receipt differs from verified checkpoint")
            for key, bins in saved["split_indices"].items():
                np.testing.assert_array_equal(bins, getattr(data, key))
            audited.append(tuning.select_policy(receipt, "ema"))
            print(f"AUDIT b2 {session} fold{fold}: verified", flush=True)
    if len(audited) != 30 or len({(r['session'],r['fold']) for r in audited}) != 30:
        raise ValueError("All30 unique b2 folds required before test")
    gate = {"status": "verified_30_b2_folds", "training_signature": signature,
            "trained_missing_folds_this_run": missing,
            "selection": "User explicitly selected b2 before these test evaluations; EMA policy unchanged",
            "protocol_note": "Separate b2-only follow-up. Does not complete or alter original78-fit b2/b4 gate.",
            "folds": audited}
    gate_path = OUTPUT / "b2_test_gate.json"
    if gate_path.exists():
        old = tuning.read_json(gate_path)
        if old["folds"] != audited or old["training_signature"] != signature:
            raise ValueError("Previously audited b2 checkpoints changed")
    else:
        write(gate_path, gate)
    print("ALL30 b2 CHECKPOINTS VERIFIED. Starting frozen EMA test evaluation.", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    baseline = tuning.read_json(tuning.REPO / "indy_loco/models/mingru_b/manifest.json")
    original_scores = {(r["session"],r["fold"]):r["test"]["r2_mean"] for r in baseline["folds"]}
    records = []
    for row in audited:
        session, fold = row["session"], row["fold"]
        test_path = OUTPUT / "test_results" / f"{session}_fold{fold}.json"
        package_path = PACKAGE / "checkpoints" / session / f"fold-{fold}.pt"
        write(OUTPUT / "status.json", {"status": "test_b2", "session": session, "fold": fold,
                                        "completed_test_folds": len(records)})
        if test_path.exists():
            record = tuning.read_json(test_path)
            if (record["source_checkpoint_sha256"] != row["checkpoint_sha256"]
                    or protocol.sha256_file(package_path) != record["sha256"]
                    or protocol.sha256_file(OUTPUT / record["prediction_file"]) != record["prediction_sha256"]):
                raise ValueError("Saved test/package integrity failure")
        else:
            data, evidence = tuning.prepare(session, fold)
            saved = tuning.original.load_training_checkpoint(tuning.REPO / row["checkpoint"])
            state = saved["weight_policies"]["ema"]["model_state"]
            model = tuning.create_model("b2").to(device).eval()
            model.load_state_dict(state, strict=True)
            exported = ModelB().to(device).eval()
            exported.load_state_dict(state, strict=True)
            probe = torch.from_numpy(protocol.rolling_batch(data.normalized_features, data.validation_bins[:16])).to(device)
            with torch.inference_mode():
                torch.testing.assert_close(exported(probe), model(probe), rtol=0, atol=0)
            prediction = protocol.predict_last(model, data.normalized_features, data.test_bins,
                                               data.target_mean, data.target_std, device, 128)
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite test predictions")
            test = protocol.metrics(data.velocity[data.test_bins], prediction)
            prediction_path = test_path.with_suffix(".npz")
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            contract.session_data.save_npz_atomic(prediction_path, prediction=prediction,
                target=data.velocity[data.test_bins], bins=data.test_bins,
                checkpoint_sha256=np.asarray(row["checkpoint_sha256"]))
            payload = {"session": session, "fold": fold, "trial": "b2", "weight_policy": "ema",
                       "model_state": {k:v.detach().cpu().clone() for k,v in state.items()},
                       "scalers": {k:torch.as_tensor(v).clone() for k,v in saved["scalers"].items()},
                       "source_checkpoint_sha256": row["checkpoint_sha256"], "training_signature": signature}
            package_path.parent.mkdir(parents=True, exist_ok=True)
            protocol.save_checkpoint_atomic(package_path, payload)
            record = {"session": session, "fold": fold, "file": package_path.relative_to(PACKAGE).as_posix(),
                      "sha256": protocol.sha256_file(package_path),
                      "source_checkpoint_sha256": row["checkpoint_sha256"], "best_epoch": row["best_epoch"],
                      "validation": row["validation"], "test": test, "preprocessing_evidence": evidence,
                      "prediction_file": prediction_path.relative_to(OUTPUT).as_posix(),
                      "prediction_sha256": protocol.sha256_file(prediction_path)}
            write(test_path, record)
            del model, exported, state, saved, probe, payload
        records.append(record)
        print(f"TEST {len(records)}/30 {session} fold{fold}: R2={record['test']['r2_mean']:.6f}", flush=True)
    values = [r["test"]["r2_mean"] for r in records]
    deltas = [r["test"]["r2_mean"]-original_scores[(r["session"],r["fold"])] for r in records]
    mean = statistics.mean(values)
    result = {"status": "complete", "folds": 30, "trial": "b2", "weight_policy": "ema",
              "validation_r2_mean": statistics.mean(r["validation"]["r2_mean"] for r in records),
              "test_r2_mean": mean, "test_r2_sample_sd": statistics.stdev(values),
              "baseline_test_r2_mean": baseline["test_r2_mean"],
              "paired_delta_mean": statistics.mean(deltas), "paired_delta_sample_sd": statistics.stdev(deltas),
              "wins_vs_original_b": sum(x>0 for x in deltas),
              "quantization_condition_met": mean > baseline["test_r2_mean"],
              "condition": "Strictly exceed exact original B mean0.7567262729008992, not rounded0.756",
              "caveat": "Single seed, correlated folds; test comparison gates quantization by explicit user request, not an unbiased independent selection.",
              "results": records}
    write(OUTPUT / "metrics.json", result)
    manifest = {"model": "mingru_b", "trial": "b2", "weight_policy": "ema", "seed": 43,
                "parameters": 374402, "fp32_weight_bytes": 1497608,
                "validation_r2_mean": result["validation_r2_mean"], "test_r2_mean": mean,
                "test_r2_sample_sd": result["test_r2_sample_sd"], "folds": records}
    write(PACKAGE / "manifest.json", manifest)
    print("B2_RESULT " + json.dumps({k:v for k,v in result.items() if k!="results"}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-quantization-if-better", action="store_true")
    args = parser.parse_args()
    with tuning.original.exclusive_run(OUTPUT):
        try:
            with ExitStack() as locks:
                locks.enter_context(tuning.original.exclusive_run(tuning.RESULTS / ".phase17_gpu"))
                locks.enter_context(tuning.original.exclusive_run(tuning.OUTPUT))
                result = audit_and_test()
            tuning.prepare.cache_clear()
            torch.cuda.empty_cache()
            if args.start_quantization_if_better and result["quantization_condition_met"]:
                destination = HERE / "results/b2_mixed_precision_v1"
                write(OUTPUT / "status.json", {"status": "quantization_running", "test_r2_mean": result["test_r2_mean"],
                                               "quantization_output": str(destination), "device": "cpu"})
                print("B2 beats original B. Starting31-plan CPU quantization sweep.", flush=True)
                with tuning.original.exclusive_run(destination):
                    for directory in ("fold_results", "calibration", "int8_weights"):
                        (destination / directory).mkdir(exist_ok=True)
                    quant.execute(SimpleNamespace(device="cpu", threads=2, package_root=PACKAGE,
                                  expected_baseline_r2=result["test_r2_mean"], resume=True), destination)
            write(OUTPUT / "status.json", {"status": "complete", "test_r2_mean": result["test_r2_mean"],
                                           "quantization_condition_met": result["quantization_condition_met"]})
        except BaseException as error:
            write(OUTPUT / "status.json", {"status": "failed", "error": str(error)})
            write(OUTPUT / "last_error.json", {"error": str(error), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
