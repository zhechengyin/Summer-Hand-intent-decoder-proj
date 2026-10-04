"""31 predeclared mixed INT8/FP32 reference plans, 30 matched folds, no training."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import statistics
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn

from indy_loco.models.mingru_b.model import ModelB
from indy_loco.experiment.phase17_architecture_comparison import data_contract as contract
from indy_loco.experiment.phase17_architecture_comparison.run import exclusive_run
from .quantization import apply_plan, ledger, placement_scenarios, plans, quantize_weight

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
protocol = contract.protocol


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


def filehash(path):
    return protocol.sha256_file(path)


def source_hashes():
    files = list(HERE.glob("*.py"))
    files += [REPO / "indy_loco/models/mingru_b/model.py"]
    files += [HERE.parent / "phase16_parameter_scaling" / name for name in ("protocol.py", "session_data.py", "protocol_lock.json")]
    files += [HERE.parent / "phase17_architecture_comparison/data_contract.py"]
    return {str(p.relative_to(REPO)): filehash(p) for p in files}


def calibrate(model, data, device):
    """Fixed 1024 evenly spaced TRAIN windows; no validation/test scale fitting."""
    locations = np.linspace(0, len(data.train_bins) - 1, min(1024, len(data.train_bins)), dtype=int)
    bins = data.train_bins[locations]
    maxima = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            maxima[name] = 0.
            def hook(module, args, name=name):
                maxima[name] = max(maxima[name], float(args[0].detach().abs().max()))
            handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            for left in range(0, len(bins), 64):
                x = torch.from_numpy(protocol.rolling_batch(data.normalized_features, bins[left:left+64])).to(device)
                model(x)
    finally:
        for handle in handles:
            handle.remove()
    if not all(np.isfinite(x) for x in maxima.values()):
        raise ValueError("Nonfinite calibration range")
    return maxima, {"split": "train", "windows": len(bins), "bin_indices_sha256": hashlib.sha256(bins.tobytes()).hexdigest(),
                    "policy": "fixed evenly spaced train windows, symmetric per-tensor input absmax, no clipping search",
                    "input_absmax": maxima}


def load_model(package, record, device):
    path = (package / record["file"]).resolve()
    if not path.is_relative_to(package.resolve()) or filehash(path) != record["sha256"]:
        raise ValueError("Checkpoint path/hash mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved["session"], saved["fold"], saved["weight_policy"]) != (record["session"], record["fold"], "ema"):
        raise ValueError("Checkpoint identity mismatch")
    model = ModelB().eval()
    model.load_state_dict(saved["model_state"], strict=True)
    if sum(p.numel() for p in model.parameters()) != 374402:
        raise ValueError("Expected the 1.50MB model B architecture")
    return model.to(device), saved


def summarize(output, rows, config, total):
    summary = []
    for plan in config["plans"]:
        selected = [r for r in rows if r["plan"] == plan["name"]]
        if not selected:
            continue
        values = [r["test"]["r2_mean"] for r in selected]
        deltas = [r["delta_test_r2"] for r in selected]
        summary.append({"plan": plan["name"], "mode": plan["mode"], "int8_groups": plan["groups"],
                        "folds": len(values), "test_r2_mean": statistics.mean(values),
                        "test_r2_sample_sd": statistics.stdev(values) if len(values)>1 else None,
                        "paired_test_r2_delta_mean": statistics.mean(deltas),
                        "paired_test_r2_delta_sample_sd": statistics.stdev(deltas) if len(deltas)>1 else None,
                        "worst_fold_test_r2_delta": min(deltas),
                        "sequential_probe_failures": sum(not r["sequential_probe_pass"] for r in selected),
                        "maximum_sequential_probe_error": max(r["sequential_probe_max_error"] for r in selected),
                        "validation_r2_mean": statistics.mean(r["validation"]["r2_mean"] for r in selected),
                        "weight_and_scale_bytes": selected[0]["memory"]["weight_and_scale_bytes"],
                        "aligned_bytes": selected[0]["memory"]["aligned_weight_and_scale_bytes"],
                        "placement_scenarios": selected[0]["placements"]})
    complete = len(rows) == total
    if complete:
        for row in summary:
            row["accuracy_size_pareto"] = not any(
                other["weight_and_scale_bytes"] <= row["weight_and_scale_bytes"]
                and other["test_r2_mean"] >= row["test_r2_mean"]
                and (other["weight_and_scale_bytes"] < row["weight_and_scale_bytes"] or other["test_r2_mean"] > row["test_r2_mean"])
                for other in summary)
    report = {"status": "complete" if complete else "partial", "completed_evaluations": len(rows),
              "expected_evaluations": total, "summary": summary,
              "test_used_for_quantization_comparison": True,
              "limitations": "Reference quantization sensitivity, not native INT8 speed. Exposed test is used for user-requested tuning, not an untouched final estimate. Correlated 30 folds, one seed. Memory budgets are hypothetical net weight budgets, not free SRAM. No MCU latency or 40ms guarantee."}
    write(output / "metrics.json", report)
    lines = ["# minGRU mixed INT8/FP32 quantization / 混合量化", "",
             f"Status: {report['status']}; {len(rows)}/{total} fold-plan evaluations.", "",
             "| Plan | Folds | Test R² mean | Paired Δ R² | Weight + scale KiB |",
             "|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['plan']} | {row['folds']} | {row['test_r2_mean']:.6f} | {row['paired_test_r2_delta_mean']:+.6f} | {row['weight_and_scale_bytes']/1024:.2f} |")
    lines += ["", report["limitations"], "", "量化误差模拟，不是 MCU 实测延迟；片上预算不含激活、工作区、栈和现有固件。",
              "test 已用于量化调参，不再是未接触的最终评估集。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def execute(args, output):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    protocol.seed_everything(43, mps=False)
    contract.verify_protocol_lock()
    manifest_path = args.package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if abs(manifest["test_r2_mean"] - args.expected_baseline_r2) > 5e-5:
        raise ValueError(f"Wrong baseline: package R2={manifest['test_r2_mean']}, requested={args.expected_baseline_r2}")
    records = manifest["folds"]
    identities = {(r["session"], r["fold"]) for r in records}
    expected = {(s, f) for s in contract.session_data.SESSION_BY_NAME for f in range(1,6)}
    if len(records) != 30 or identities != expected:
        raise ValueError("Require all 30 unique folds")
    config = {"package_root": str(args.package_root.resolve()), "manifest_sha256": filehash(manifest_path),
              "expected_baseline_r2": args.expected_baseline_r2, "plans": plans(),
              "device": str(device), "threads": args.threads, "source_sha256": source_hashes(),
              "torch": torch.__version__, "calibration_train_windows": 1024,
              "quantizer": "per-output-channel symmetric signed INT8 [-127,127], round-to-nearest-even",
              "bias_norm_nonlinearity_recurrence": "FP32",
              "evaluation": "parallel recurrence full datasets; sequential parity probe per fold/plan",
              "test_used_for_quantization_tuning": True}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_file = output / "config.json"
    if config_file.exists():
        if not args.resume or json.loads(config_file.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing configuration differs or --resume not provided")
    else:
        write(config_file, config)
    contract.configure_paths(contract.INDY / "data", contract.default_gui_root(), output / ".cache")
    rows = []
    total = len(records) * len(config["plans"])
    for session in contract.session_data.SESSION_BY_NAME:
        raw = contract.load_session(session, contract.REFERENCE.parent / ".cache/session_inputs")
        for record in sorted((r for r in records if r["session"] == session), key=lambda r:r["fold"]):
            fold = record["fold"]
            data, evidence = contract.prepare_verified_fold(raw, fold, contract.load_reference(session, fold))
            if evidence != record["preprocessing_evidence"]:
                raise ValueError("Quantization preprocessing differs from source model")
            model, saved = load_model(args.package_root, record, device)
            for field, key in (("target_mean", "target_mean"), ("target_std", "target_std")):
                np.testing.assert_array_equal(getattr(data, field), np.asarray(saved["scalers"][key]).reshape(-1))
            maxima, calibration = calibrate(model, data, device)
            calibration_path = output / "calibration" / f"{session}_fold{fold}.json"
            if calibration_path.exists() and json.loads(calibration_path.read_text()) != calibration:
                raise ValueError("Calibration drift on resume")
            write(calibration_path, calibration)
            # A universal packed INT8 weight bundle plus source checkpoint reference:
            # plans select which tensors use these values versus original FP32.
            bundle_path = output / "int8_weights" / f"{session}_fold{fold}.pt"
            if not bundle_path.exists():
                bundle = {}
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv1d)):
                        q, scale = quantize_weight(module.weight)
                        bundle[name] = {"weight": q.cpu(), "scale": scale.cpu(), "input_absmax": maxima[name]}
                protocol.save_checkpoint_atomic(bundle_path, {"source_sha256": record["sha256"], "signature": signature,
                                                              "operators": bundle})
            base = None
            for plan in config["plans"]:
                identity = f"{session}_fold{fold}_{plan['name']}"
                path = output / "fold_results" / (identity + ".json")
                if source_hashes() != config["source_sha256"]:
                    raise ValueError("Source changed during run")
                if path.exists():
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if row["signature"] != signature or row["source_sha256"] != record["sha256"]:
                        raise ValueError("Saved fold/plan identity changed")
                    if plan["name"] == "fp32":
                        base = row["test"]["r2_mean"]
                    rows.append(row)
                    continue
                write(output / "status.json", {"status": "evaluating", "current": identity,
                                                "completed": len(rows), "total": total, "device": str(device)})
                candidate = apply_plan(model, plan, maxima).to(device).eval()
                memory = ledger(model, plan)
                started = time.perf_counter()
                scores = {}
                for split in ("validation", "test"):
                    scores[split] = protocol.evaluate_last(candidate, data.normalized_features, data.velocity,
                        getattr(data, split + "_bins"), data.target_mean, data.target_std, device, 128)
                if plan["name"] == "fp32":
                    base = scores["test"]["r2_mean"]
                    if abs(base - record["test"]["r2_mean"]) > 1e-4:
                        raise ValueError(f"FP32 test reproduction failed: {base} vs {record['test']['r2_mean']}")
                probe = torch.from_numpy(protocol.rolling_batch(data.normalized_features, data.validation_bins[:8])).to(device)
                with torch.inference_mode():
                    parallel, sequential = candidate(probe), candidate.forward_sequential(probe)
                    parity_error = float((parallel-sequential).abs().max())
                    # Crossing an activation rounding threshold in W8A8 can cause
                    # larger sequential differences; record, do not claim equivalence.
                    parity_pass = bool(torch.allclose(parallel, sequential, atol=2e-4, rtol=2e-4))
                row = {"session": session, "fold": fold, "plan": plan["name"], "signature": signature,
                       "source_sha256": record["sha256"], **scores, "delta_test_r2": scores["test"]["r2_mean"]-base,
                       "memory": memory, "placements": placement_scenarios(memory),
                       "reference_evaluation_seconds": time.perf_counter()-started,
                       "sequential_probe_max_error": parity_error, "sequential_probe_pass": parity_pass,
                       "mcu_latency_ms": None, "native_int8_latency_measured": False}
                write(path, row)
                rows.append(row)
                summarize(output, rows, config, total)
                print(f"{len(rows)}/{total} {identity}: test R2={scores['test']['r2_mean']:.6f} "
                      f"delta={row['delta_test_r2']:+.6f} weights={memory['weight_and_scale_bytes']/1024:.1f}KiB", flush=True)
                del candidate, probe, parallel, sequential
            del model, saved, data
        del raw
    summarize(output, rows, config, total)
    write(output / "status.json", {"status": "complete", "completed": len(rows), "total": total})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--expected-baseline-r2", type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-name", default="mixed_precision_v1")
    args = parser.parse_args()
    if not args.output_name.replace("_", "").replace("-", "").isalnum() or not 1 <= args.threads <= 8:
        parser.error("Use a simple output name and 1..8 CPU threads")
    output = HERE / "results" / args.output_name
    with ExitStack() as locks:
        if args.device == "cuda":
            locks.enter_context(exclusive_run(HERE.parent / "phase17_architecture_comparison/results/.phase17_gpu"))
        locks.enter_context(exclusive_run(output))
        for folder in ("fold_results", "calibration", "int8_weights"):
            (output / folder).mkdir(exist_ok=True)
        try:
            execute(args, output)
        except BaseException as error:
            write(output / "status.json", {"status": "failed", "error": str(error)})
            write(output / "last_error.json", {"error": str(error), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
