"""Evaluate frozen 96/128/256 checkpoints and canonical Midsize on identical test bins."""
import argparse
import hashlib
import json
import os
import statistics
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from indy_loco.experiment.phase16_parameter_scaling import protocol
from indy_loco.experiment.phase16_parameter_scaling.model import Architecture, ScaledTCNGRU
from indy_loco.experiment.phase17_architecture_comparison import data_contract as contract
from indy_loco.experiment.phase17_architecture_comparison.run import exclusive_run

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SWEEP = HERE.parent / "results/size_sweep_v1"
OUTPUT = HERE.parent / "results/size_sweep_test_v1"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write(p, value):
    protocol.write_json_atomic(p, value)


def run(device):
    config = read(SWEEP / "config.json")
    metrics = read(SWEEP / "metrics.json")
    if metrics["status"] != "complete" or metrics["completed_jobs"] != 90:
        raise ValueError("All validation-selected folds must be complete before opening test")
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    for relative, expected in config["source_sha256"].items():
        if protocol.sha256_file(REPO / relative) != expected:
            raise ValueError(f"Frozen training source changed: {relative}")
    contract.verify_protocol_lock()
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for same-backend comparison")
    torch.set_num_threads(2 if device.type == "cpu" else 4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    protocol.seed_everything(43, mps=False)
    manifest = {"training_signature": signature, "training_metrics_sha256": protocol.sha256_file(SWEEP / "metrics.json"),
                "evaluation_source_sha256": protocol.sha256_file(Path(__file__)),
                "torch": torch.__version__, "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                "precision": "FP32, TF32 disabled", "widths": [64, 96, 128, 256],
                "selection_frozen_before_test": True, "validation_ranking": metrics["validation_ranking"]}
    if (OUTPUT / "config.json").exists() and read(OUTPUT / "config.json") != manifest:
        raise ValueError("Existing evaluation configuration changed")
    write(OUTPUT / "config.json", manifest)
    contract.configure_paths(contract.INDY / "data", contract.default_gui_root(), OUTPUT / ".cache")
    results = []
    for session in config["sessions"]:
        raw = contract.load_session(session, contract.REFERENCE.parent / ".cache/session_inputs")
        for fold in config["folds"]:
            refpath = contract.reference_path(session, fold)
            if protocol.sha256_file(refpath) != config["references"][f"{session}_fold{fold}"]:
                raise ValueError("Canonical reference changed")
            reference = contract.load_reference(session, fold)
            data, evidence = contract.prepare_verified_fold(raw, fold, reference)
            for width in (64, 96, 128, 256):
                name = f"w{width}_{session}_fold{fold}"
                architecture = Architecture(width, 3, (1, 2, 4, 8), width, 1)
                if width == 64:
                    checkpoint, checkpoint_path = reference, refpath
                else:
                    row = read(SWEEP / "fold_results" / f"{name}.json")
                    checkpoint_path = SWEEP / row["checkpoint"]
                    if (row["signature"] != signature or row["preprocessing_evidence"] != evidence
                            or protocol.sha256_file(checkpoint_path) != row["checkpoint_sha256"]):
                        raise ValueError(f"Checkpoint or preprocessing mismatch: {name}")
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    if (checkpoint["signature"] != signature or checkpoint["session"] != session
                            or checkpoint["fold"] != fold or checkpoint["architecture"] != architecture.metadata()
                            or checkpoint["preprocessing_evidence"] != evidence
                            or checkpoint["best_epoch"] != row["best_epoch"]):
                        raise ValueError(f"Checkpoint metadata mismatch: {name}")
                digest = protocol.sha256_file(checkpoint_path)
                result_path = OUTPUT / "fold_results" / f"{name}.json"
                prediction_path = OUTPUT / "predictions" / f"{name}.npz"
                if result_path.exists():
                    result = read(result_path)
                    if (result["checkpoint_sha256"] != digest or result["preprocessing_evidence"] != evidence
                            or result["prediction_sha256"] != protocol.sha256_file(prediction_path)):
                        raise ValueError(f"Saved test evidence changed: {name}")
                else:
                    write(OUTPUT / "status.json", {"status": "evaluating", "current": name, "completed": len(results), "expected": 120})
                    model = ScaledTCNGRU(architecture)
                    model.load_state_dict(checkpoint["model_state"], strict=True)
                    model.to(device).eval()
                    prediction = protocol.predict_last(model, data.normalized_features, data.test_bins,
                        data.target_mean, data.target_std, device, 128)
                    target = data.velocity[data.test_bins]
                    score = protocol.metrics(target, prediction)
                    score["normalized_loss"] = float(np.mean(((target - prediction) / data.target_std) ** 2))
                    np.savez_compressed(prediction_path, bins=data.test_bins, target=target, prediction=prediction)
                    result = {"width": width, "session": session, "fold": fold, "test": score,
                              "checkpoint_sha256": digest, "prediction_sha256": protocol.sha256_file(prediction_path),
                              "preprocessing_evidence": evidence}
                    write(result_path, result)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                results.append(result)
                print(f"{len(results)}/120 {name}: test R2={result['test']['r2_mean']:.6f}", flush=True)
            del data, reference, checkpoint
        del raw
    summaries = {}
    baseline = {(r["session"], r["fold"]): r["test"]["r2_mean"] for r in results if r["width"] == 64}
    for width in (64, 96, 128, 256):
        rows = [r for r in results if r["width"] == width]
        values = [r["test"]["r2_mean"] for r in rows]
        delta = [r["test"]["r2_mean"] - baseline[(r["session"], r["fold"])] for r in rows]
        summaries[str(width)] = {"folds": len(rows), "test_r2_mean": statistics.mean(values),
            "test_r2_sample_sd": statistics.stdev(values), "paired_delta_mean": statistics.mean(delta),
            "paired_delta_sample_sd": statistics.stdev(delta), "wins": sum(d > 0 for d in delta),
            "losses": sum(d < 0 for d in delta),
            "per_session": {s: statistics.mean(r["test"]["r2_mean"] for r in rows if r["session"] == s)
                            for s in config["sessions"]}}
    report = {"status": "complete", "evaluated_checkpoints": 120, "full_30fold_per_width": True,
              "summary": summaries, "results": results, "test_evaluated": True,
              "caveat": "Canonical warm-started Midsize re-evaluated on the same evaluation backend. Wider models received additional validation-selected optimization. This does not isolate width from extra training. Test scores did not choose checkpoints or change the frozen validation ranking."}
    write(OUTPUT / "metrics.json", report)
    lines = ["# Frozen width-sweep test evaluation / 宽度扫描 test 结果", "",
             "| Width | Test R² mean ± sample SD | Paired Δ vs Midsize | Wins / 30 |",
             "|---|---:|---:|---:|"]
    for w, row in summaries.items():
        lines.append(f"| {w} | {row['test_r2_mean']:.6f} ± {row['test_r2_sample_sd']:.6f} | {row['paired_delta_mean']:+.6f} | {row['wins']} |")
    lines += ["", report["caveat"], "", "基线和扩宽模型均有 warm-start；扩宽模型还经过额外训练，因此不能把差值完全归因于宽度。"]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write(OUTPUT / "status.json", {"status": "complete", "completed": 120, "expected": 120})
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    lock = HERE.parent.parent / "phase17_architecture_comparison/results/.phase17_gpu" if args.device == "cuda" else OUTPUT
    with exclusive_run(lock):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name in ("fold_results", "predictions"):
            (OUTPUT / name).mkdir(exist_ok=True)
        try:
            run(device)
        except BaseException as error:
            write(OUTPUT / "last_error.json", {"error": str(error), "traceback": traceback.format_exc()})
            write(OUTPUT / "status.json", {"status": "failed", "error": str(error)})
            raise
