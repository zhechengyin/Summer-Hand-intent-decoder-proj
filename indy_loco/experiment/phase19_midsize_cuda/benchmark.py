"""Phase19: gated, single-seed Midsize CUDA training-time ablations.

Run explicitly with python -m; importing this module never starts an experiment.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from indy_loco.experiment.phase17_architecture_comparison import data_contract as contract
from indy_loco.experiment.phase17_architecture_comparison.run import exclusive_run
from .model import VARIANTS, create_model, optimizer_for
from .runtime import runtime
from .verify import TOLERANCES, check_baseline, check_variant

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SHARED_GPU_LOCK = HERE.parent / "phase17_architecture_comparison/results/.phase17_gpu"
protocol = contract.protocol
SESSION = "indy_20160622_01"
SEED = 43
BATCH_SIZE = 128


def write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def fingerprints():
    files = list(HERE.glob("*.py")) + list(HERE.glob("*.cu"))
    for folder, names in (
        ("phase16_parameter_scaling", ("model.py", "protocol.py", "session_data.py", "protocol_lock.json")),
        ("phase17_architecture_comparison", ("data_contract.py", "run.py")),
    ):
        files.extend(HERE.parent / folder / name for name in names)
    files.append(REPO / "indy_loco/models/midsize/model.py")
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def cuda_setup():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. This benchmark has no CPU fallback.")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    protocol.seed_everything(SEED, mps=False)


def train_short(variant, state, data, epochs, output, repeat):
    protocol.seed_everything(SEED, mps=False)
    model = create_model(variant, state).cuda().train()
    warm = torch.from_numpy(protocol.rolling_batch(
        data.normalized_features, data.train_bins[:BATCH_SIZE])).cuda()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        model(warm)[:, -1].square().mean().backward()
    torch.cuda.synchronize()
    warmup = time.perf_counter() - start
    # No optimizer update during warm-up. Still restore state and all RNGs.
    model.load_state_dict(state, strict=True)
    model.zero_grad(set_to_none=True)
    del warm
    protocol.seed_everything(SEED, mps=False)
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    rng = np.random.default_rng(SEED)
    targets = ((data.velocity - data.target_mean) / data.target_std).astype(np.float32)
    best_loss, best_epoch, best_state = float("inf"), None, None
    history = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        model.train()
        torch.cuda.synchronize()
        started = time.perf_counter()
        order = rng.permutation(data.train_bins)
        error_sum, count = 0.0, 0
        gradient_max = 0.0
        for left in range(0, len(order), BATCH_SIZE):
            bins = order[left:left + BATCH_SIZE]
            inputs = torch.from_numpy(protocol.rolling_batch(data.normalized_features, bins)).cuda()
            target = torch.from_numpy(targets[bins]).cuda()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)[:, -1]
            loss = (prediction - target).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss in {variant}")
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                           error_if_nonfinite=True))
            optimizer.step()
            error_sum += float(loss.detach()) * len(bins)
            count += len(bins)
            gradient_max = max(gradient_max, gradient)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        validation_start = time.perf_counter()
        validation = protocol.evaluate_last(
            model, data.normalized_features, data.velocity, data.validation_bins,
            data.target_mean, data.target_std, torch.device("cuda"), BATCH_SIZE)
        torch.cuda.synchronize()
        validation_seconds = time.perf_counter() - validation_start
        if validation["normalized_loss"] < best_loss:
            best_loss = validation["normalized_loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()
        row = {"epoch": epoch, "optimization_loss": error_sum / count,
               "gradient_max": gradient_max, "training_seconds": train_seconds,
               "validation_seconds": validation_seconds, "validation": validation,
               "compute_and_selection_seconds": time.perf_counter() - started}
        history.append(row)
        # Same per-epoch log work for all arms, including the baseline.
        write(output / f"repeat{repeat}_{variant}.epochs.json", history)
        print(f"repeat {repeat} {variant} epoch {epoch}/{epochs}: "
              f"train={train_seconds:.3f}s val_R2={validation['r2_mean']:.6f}", flush=True)
        row["end_to_end_seconds"] = time.perf_counter() - started
    result = {"repeat": repeat, "variant": variant, "seed": SEED,
              "warmup_seconds": warmup, "epochs": history,
              "best_validation_epoch": best_epoch, "best_validation_loss": best_loss,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    write(output / f"repeat{repeat}_{variant}.json", result)
    # Timing evidence only: do not replace or promote any canonical model.
    del model, optimizer, scheduler, best_state
    torch.cuda.empty_cache()
    return result


def summarize(results, checks, config):
    per_variant = {}
    for variant in config["eligible_variants"]:
        runs = [r for r in results if r["variant"] == variant]
        means = [statistics.mean(e["end_to_end_seconds"] for e in r["epochs"]) for r in runs]
        per_variant[variant] = {
            "full_epoch_seconds_mean": statistics.mean(means),
            "repeat_mean_sample_sd": statistics.stdev(means) if len(means) > 1 else None,
            "training_seconds_mean": statistics.mean(
                e["training_seconds"] for r in runs for e in r["epochs"]),
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in runs),
            "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in runs),
            "last_epoch_validation_r2": [r["epochs"][-1]["validation"]["r2_mean"] for r in runs],
        }
    baseline = per_variant["pytorch"]
    for variant, row in per_variant.items():
        if variant == "pytorch":
            continue
        pairs = [(next(r for r in results if r["variant"] == "pytorch" and r["repeat"] == i),
                  next(r for r in results if r["variant"] == variant and r["repeat"] == i))
                 for i in range(1, config["repeats"] + 1)]
        speedups = [statistics.mean(e["end_to_end_seconds"] for e in a["epochs"]) /
                    statistics.mean(e["end_to_end_seconds"] for e in b["epochs"])
                    for a, b in pairs]
        delta = max(abs(x["validation"]["r2_mean"] - y["validation"]["r2_mean"])
                    for a, b in pairs for x, y in zip(a["epochs"], b["epochs"], strict=True))
        speedup = baseline["full_epoch_seconds_mean"] / row["full_epoch_seconds_mean"]
        row.update(speedup_vs_pytorch=speedup, time_reduction_fraction=1 - 1 / speedup,
                   paired_repeat_speedups=speedups, max_epoch_validation_r2_difference=delta,
                   adoption_candidate=(config["repeats"] >= 3 and config["epochs"] >= 3
                                       and speedup >= 1 / 0.9 and min(speedups) > 1 and delta <= .001))
    return {
        "status": "complete", "variants": per_variant,
        "rejected_variants": {k: v["error"] for k, v in checks.items() if v["status"] == "rejected"},
        "test_evaluated": False,
        "limitations": "One session/fold/seed, warm-started from canonical fold1. Repeats measure timing, not independent accuracy seeds. Short-run validation does not establish final test R2 or STM32 speed. cuDNN may beat custom GRU. No automatic adoption.",
    }


def markdown_report(summary):
    lines = ["# Phase19 Midsize CUDA comparison / 对比结果", "",
             "Same FP32 model and data; failed numerical candidates are excluded from timing.",
             "相同 FP32 模型和数据；未通过数值检查的候选不进入计时。", "",
             "| Variant | Full epoch seconds | Training seconds | Speedup | Max validation R² delta | Candidate |",
             "|---|---:|---:|---:|---:|---|"]
    for name, row in summary["variants"].items():
        lines.append(f"| {name} | {row['full_epoch_seconds_mean']:.4f} | "
                     f"{row['training_seconds_mean']:.4f} | {row.get('speedup_vs_pytorch', 1):.3f}x | "
                     f"{row.get('max_epoch_validation_r2_difference', 0):.6f} | "
                     f"{row.get('adoption_candidate', False)} |")
    lines += ["", "## Rejected numerical candidates / 数值检查未通过", ""]
    for name, error in summary["rejected_variants"].items():
        lines.append(f"- {name}: {error}")
    if not summary["rejected_variants"]:
        lines.append("None / 无")
    lines += ["", summary["limitations"], "",
              "单 session、第1折、单 seed 的短训练计时；复用该折 Midsize 初始权重。",
              "不评估 test，不代表最终 R² 或 STM32 性能，不自动替换原模型。", ""]
    return "\n".join(lines)


def execute(args, output):
    cuda_setup()
    contract.verify_protocol_lock()
    contract.configure_paths(args.data_root, args.gui_root, output / ".cache")
    reference = contract.load_reference(SESSION, 1)
    checkpoint_path = contract.reference_path(SESSION, 1)
    session = contract.load_session(SESSION, args.indy_cache_root)
    data, evidence = contract.prepare_verified_fold(session, 1, reference)
    state = reference["model_state"]
    del session, reference
    config = {
        "seed": SEED, "session": SESSION, "fold": 1, "epochs": args.epochs,
        "repeats": args.repeats, "requested_variants": args.variants,
        "batch_size": BATCH_SIZE, "parameters": 86978,
        "initialization": "same_canonical_Midsize_fold1_checkpoint_all_arms",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": protocol.sha256_file(checkpoint_path),
        "preprocessing_evidence": evidence, "source_sha256": fingerprints(),
        "hyperparameters": {"learning_rate_gru_head": 3e-4, "encoder_lr_scale": .25,
                            "weight_decay": .025, "gradient_clip": 1.0,
                            "scheduler_T_max": 20, "channel_dropout": .20,
                            "model_dropout": .10, "early_stopping": False},
        "precision": "FP32, no TF32 or AMP; NVRTC fast math disabled",
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(), "cpu_threads": 4,
        "numerical_tolerances": TOLERANCES, "test_evaluated": False,
        "verify_only": args.verify_only,
    }
    write(output / "config.json", config)
    write(output / "status.json", {"status": "compiling"})
    print("Compiling isolated Phase19 CUDA kernels... / 编译 Phase19 内核", flush=True)
    started = time.perf_counter()
    compiled = runtime(torch.device("cuda"))
    write(output / "compilation.json", {**compiled.metadata,
                                        "compile_and_load_seconds": time.perf_counter() - started})
    real_inputs = torch.from_numpy(protocol.rolling_batch(
        data.normalized_features, data.train_bins[:BATCH_SIZE])).cuda()
    real_targets = torch.from_numpy(((data.velocity[data.train_bins[:BATCH_SIZE]]
                                     - data.target_mean) / data.target_std).astype(np.float32)).cuda()
    write(output / "baseline_verification.json", check_baseline(state, real_inputs))
    checks = {}
    for variant in args.variants:
        if variant == "pytorch":
            continue
        write(output / "status.json", {"status": "verifying", "variant": variant})
        print(f"Checking {variant}: output, gradients and AdamW...", flush=True)
        checks[variant] = check_variant(variant, state, real_inputs, real_targets)
        write(output / "verification.json", checks)
        print(f"{variant}: {checks[variant]['status']}", flush=True)
        if checks[variant]["status"] == "rejected":
            print(checks[variant]["error"], flush=True)
        torch.cuda.empty_cache()
    del real_inputs, real_targets
    if args.verify_only:
        write(output / "status.json", {"status": "verification_complete", "training_started": False,
                                        "results": {k: v["status"] for k, v in checks.items()}})
        return
    eligible = ["pytorch"] + [v for v in args.variants if v != "pytorch" and checks[v]["status"] == "passed"]
    config["eligible_variants"] = eligible
    orders = []
    for index in range(args.repeats):
        shift = (index * 2) % len(eligible)
        order = eligible[shift:] + eligible[:shift]
        orders.append(list(reversed(order)) if index % 2 else order)
    config["order"] = orders
    write(output / "config.json", config)
    if len(eligible) == 1:
        write(output / "status.json", {"status": "no_candidate_passed", "training_started": False})
        print("No candidate passed; no timing training started. See verification.json.", flush=True)
        return
    results = []
    for repeat, order in enumerate(orders, 1):
        for variant in order:
            if fingerprints() != config["source_sha256"]:
                raise RuntimeError("Source changed during benchmark; do not mix timing evidence")
            write(output / "status.json", {"status": "training", "repeat": repeat, "variant": variant})
            results.append(train_short(variant, state, data, args.epochs, output, repeat))
    summary = summarize(results, checks, config)
    write(output / "summary.json", summary)
    (output / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    write(output / "status.json", {"status": "complete", "test_evaluated": False})
    print(markdown_report(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--verify-only", action="store_true", help="Compile and check equivalence; no training")
    parser.add_argument("--output-name", help="New directory name under Phase19/results; never overwritten")
    parser.add_argument("--data-root", type=Path, default=contract.INDY / "data")
    parser.add_argument("--gui-root", type=Path, default=contract.default_gui_root())
    parser.add_argument("--indy-cache-root", type=Path, default=contract.REFERENCE.parent / ".cache/session_inputs")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20 or not 1 <= args.repeats <= 6:
        parser.error("Use 1..20 epochs and 1..6 repeats for this short benchmark")
    args.variants = list(dict.fromkeys(["pytorch", *args.variants]))
    if len(args.variants) < 2:
        parser.error("Select at least one custom candidate")
    name = args.output_name or (datetime.now().strftime("run_%Y%m%d_%H%M%S_") + str(os.getpid()))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        parser.error("output-name must be one simple directory name")
    output = HERE / "results" / name
    # Shares the advisory OS lock used by Phase17/18; no overlapping cooperating runs.
    with exclusive_run(SHARED_GPU_LOCK):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(exist_ok=False)
        print(f"Phase19 output / 结果目录: {output}", flush=True)
        try:
            write(output / "status.json", {"status": "preparing"})
            execute(args, output)
        except BaseException as error:
            payload = {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       "error": str(error), "traceback": traceback.format_exc()}
            write(output / "last_error.json", payload)
            write(output / "status.json", payload)
            raise


if __name__ == "__main__":
    main()
