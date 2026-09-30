"""Five-epoch, single-seed eager PyTorch vs custom minGRU CUDA benchmark."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
import numpy as np
import torch

from indy_loco.experiment.phase18_large_mingru.ab_study import train as ab
from indy_loco.experiment.phase18_large_mingru.cuda_equivalence.kernel import (
    compile_ptx,
    install,
    runtime,
)
from indy_loco.experiment.phase18_large_mingru.cuda_equivalence.verify import (
    cpu_checks,
    cuda_checks,
)

OUTPUT = HERE.parent / "results/cuda_equivalence_v3"
SESSION = "indy_20160622_01"
TRIAL = ab.plan.TRIALS["t00"]


def write(path, value):
    """Retry transient Windows replacement failures without touching old helpers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def create_model():
    ab.protocol.seed_everything(43, mps=False)
    return ab.create_model("mingru_b", TRIAL)


def fit_short(variant, data, epochs, destination):
    model = create_model().cuda()
    if variant == "custom_cuda":
        install(model)
    initial = copy.deepcopy(model.state_dict())
    # Warm-up performs no optimizer steps; restore buffers, gradients and RNG.
    warm = torch.from_numpy(
        ab.protocol.rolling_batch(data.normalized_features, data.train_bins[:128])
    ).cuda()
    start = time.perf_counter()
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        model(warm).square().mean().backward()
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - start
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    del initial, warm
    ab.protocol.seed_everything(43, mps=False)
    optimizer = torch.optim.AdamW(
        ab.optimizer_groups(model, "mingru_b", TRIAL),
        weight_decay=TRIAL["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 60)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    targets = ((data.velocity - data.target_mean) / data.target_std).astype(np.float32)
    rng = np.random.default_rng(43)
    best = {policy: {"loss": float("inf")} for policy in ("raw", "ema")}
    history = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        torch.cuda.synchronize()
        beginning = time.perf_counter()
        model.train()
        order = rng.permutation(data.train_bins)
        error_sum, count, gradient_max = 0.0, 0, 0.0
        for left in range(0, len(order), 128):
            bins = order[left : left + 128]
            inputs = torch.from_numpy(
                ab.protocol.rolling_batch(data.normalized_features, bins)
            ).cuda()
            target = torch.from_numpy(targets[bins]).cuda()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)[:, -1]
            loss = (prediction - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
            )
            optimizer.step()
            ab.update_ema(ema, model)
            error_sum += float(loss.detach()) * len(bins)
            count += len(bins)
            gradient_max = max(gradient_max, gradient)
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - beginning
        scores, probes = {}, {}
        for policy, candidate in (("raw", model), ("ema", ema)):
            scores[policy] = ab.protocol.evaluate_last(
                candidate,
                data.normalized_features,
                data.velocity,
                data.validation_bins,
                data.target_mean,
                data.target_std,
                torch.device("cuda"),
                128,
            )
            probes[policy] = ab.evaluate_training_probe(
                candidate, data, torch.device("cuda")
            )
            if scores[policy]["normalized_loss"] < best[policy]["loss"]:
                best[policy] = {
                    "loss": scores[policy]["normalized_loss"],
                    "state": {
                        k: v.detach().cpu().clone()
                        for k, v in candidate.state_dict().items()
                    },
                }
        scheduler.step()
        torch.cuda.synchronize()
        row = {
            "epoch": epoch,
            "training_seconds": training_seconds,
            "epoch_compute_seconds": time.perf_counter() - beginning,
            "optimization_loss": error_sum / count,
            "gradient_max": gradient_max,
            "validation": scores,
            "training_probe": probes,
        }
        history.append(row)
        # Equivalent logging/checkpoint-selection work in both arms. Report its
        # I/O separately from the GPU-synchronized compute interval.
        io_start = time.perf_counter()
        write(destination.with_suffix(".epochs.json"), history)
        print(
            f"{variant} epoch {epoch}: train={training_seconds:.3f}s full_compute={row['epoch_compute_seconds']:.3f}s ema_R2={scores['ema']['r2_mean']:.6f}",
            flush=True,
        )
        row["logging_seconds"] = time.perf_counter() - io_start
        row["end_to_end_seconds"] = time.perf_counter() - beginning
    result = {
        "variant": variant,
        "seed": 43,
        "epochs": history,
        "warmup_seconds": warmup_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    write(destination, result)
    del model, ema, optimizer, best
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.epochs != 5 or args.repeats != 3:
        parser.error(
            "Prespecified benchmark is five epochs, three timing repeats, seed43"
        )
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.cpu_check:
        result = cpu_checks()
        write(OUTPUT / "cpu_checks.json", result)
        print(json.dumps(result))
        return
    if args.compile_only:
        started = time.perf_counter()
        ptx, metadata = compile_ptx()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "mingru_compute89.ptx").write_bytes(ptx)
        metadata.update(
            status="compiled_only",
            seconds=time.perf_counter() - started,
            cuda_execution_verified=False,
        )
        write(OUTPUT / "compilation.json", metadata)
        print(json.dumps(metadata))
        return
    if not torch.cuda.is_available():
        result = {
            "status": "blocked",
            "reason": "Fresh torch process sees no CUDA GPU",
            "torch": torch.__version__,
            "cuda_devices": torch.cuda.device_count(),
            "speedup_measured": False,
        }
        write(OUTPUT / "status.json", result)
        raise RuntimeError(result["reason"])
    with ExitStack() as locks:
        locks.enter_context(ab.original.exclusive_run(ab.RESULTS / ".phase17_gpu"))
        locks.enter_context(ab.original.exclusive_run(OUTPUT))
        if (OUTPUT / "summary.json").exists():
            raise FileExistsError(
                "Completed evidence exists; do not overwrite benchmark"
            )
        ab.contract.verify_protocol_lock()
        ab.contract.configure_paths(
            ab.contract.INDY / "data", ab.contract.default_gui_root(), OUTPUT / ".cache"
        )
        # Verify frozen model/training sources without reading test results.
        old = ab.read_json(ab.OUTPUT / "config.json")
        for relative, expected in old["code_sha256"].items():
            if (
                hashlib.sha256(
                    (REPO / relative).read_text(encoding="utf-8").encode()
                ).hexdigest()
                != expected
            ):
                raise ValueError(f"Frozen code changed: {relative}")
        data, evidence = ab.prepare(SESSION, 1)
        code = {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in HERE.iterdir()
            if f.suffix in (".py", ".cu")
        }
        config = {
            "seed": 43,
            "session": SESSION,
            "fold": 1,
            "epochs": 5,
            "repeats": 3,
            "architecture": "unchanged_mingru_b",
            "hyperparameters": TRIAL,
            "fp32": True,
            "tf32": False,
            "data_preprocessing": evidence,
            "source_sha256": code,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "scheduler_T_max": 60,
            "test_used": False,
            "only_replacement": "candidate/gate, log scan and backward",
            "order": [
                ["pytorch", "custom_cuda"],
                ["custom_cuda", "pytorch"],
                ["pytorch", "custom_cuda"],
            ],
        }
        if (OUTPUT / "config.json").exists() and ab.read_json(
            OUTPUT / "config.json"
        ) != config:
            raise ValueError("Existing benchmark configuration changed")
        write(OUTPUT / "config.json", config)
        started = time.perf_counter()
        runtime(torch.device("cuda"))
        compilation_seconds = time.perf_counter() - started
        checks = cuda_checks(create_model)
        checks["initial_compilation_load_seconds"] = compilation_seconds
        write(OUTPUT / "cuda_checks.json", checks)
        results = []
        for repeat, order in enumerate(config["order"], 1):
            for variant in order:
                path = OUTPUT / f"repeat{repeat}_{variant}.json"
                if path.exists():
                    raise FileExistsError(
                        "Partial timing run exists; inspect before choosing a new run directory"
                    )
                write(
                    OUTPUT / "status.json",
                    {"status": "running", "repeat": repeat, "variant": variant},
                )
                result = fit_short(variant, data, 5, path)
                result["repeat"] = repeat
                write(path, result)
                results.append(result)
        summary = {}
        for variant in ("pytorch", "custom_cuda"):
            runs = [r for r in results if r["variant"] == variant]
            repeat_means = [
                statistics.mean(e["end_to_end_seconds"] for e in r["epochs"])
                for r in runs
            ]
            summary[variant] = {
                "seconds_per_epoch": statistics.mean(repeat_means),
                "repeat_mean_sample_sd": statistics.stdev(repeat_means),
                "training_seconds_per_epoch": statistics.mean(
                    e["training_seconds"] for r in runs for e in r["epochs"]
                ),
                "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in runs),
            }
        speedup = (
            summary["pytorch"]["seconds_per_epoch"]
            / summary["custom_cuda"]["seconds_per_epoch"]
        )
        pairs = [
            (
                next(
                    r
                    for r in results
                    if r["variant"] == "pytorch" and r["repeat"] == repeat
                ),
                next(
                    r
                    for r in results
                    if r["variant"] == "custom_cuda" and r["repeat"] == repeat
                ),
            )
            for repeat in range(1, 4)
        ]
        delta = max(
            abs(a["validation"][policy]["r2_mean"] - b["validation"][policy]["r2_mean"])
            for x, y in pairs
            for a, b in zip(x["epochs"], y["epochs"], strict=True)
            for policy in ("raw", "ema")
        )
        paired_speedups = [
            statistics.mean(e["end_to_end_seconds"] for e in a["epochs"])
            / statistics.mean(e["end_to_end_seconds"] for e in b["epochs"])
            for a, b in pairs
        ]
        report = {
            "status": "complete",
            "summary": summary,
            "speedup": speedup,
            "paired_repeat_speedups": paired_speedups,
            "wall_time_reduction": 1 - 1 / speedup,
            "max_epoch_validation_r2_difference": delta,
            "short_run_equivalence_threshold": 0.001,
            "short_run_equivalence_pass": delta <= 0.001,
            "adoption_candidate": delta <= 0.001
            and speedup >= 1 / 0.9
            and min(paired_speedups) > 1,
            "limitations": "Single session/fold/seed; timing repeats are not independent accuracy seeds. Five epochs do not prove final R2 equivalence. CUDA higher-order gradients unsupported.",
            "config": config,
        }
        write(OUTPUT / "summary.json", report)
        write(OUTPUT / "status.json", {"status": "complete", "speedup": speedup})
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write(
            OUTPUT / "last_error.json",
            {"error": str(error), "traceback": traceback.format_exc()},
        )
        raise
