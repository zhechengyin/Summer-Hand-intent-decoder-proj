"""Validation-only 96/96, 128/128, 256/256 Midsize width sweep."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from indy_loco.experiment.phase16_parameter_scaling import fast_training, protocol
from indy_loco.experiment.phase16_parameter_scaling.model import Architecture, ScaledTCNGRU, parameter_report
from indy_loco.experiment.phase16_parameter_scaling.stopping import StopCriterion
from indy_loco.experiment.phase16_parameter_scaling.transfer import transfer_midsize_weights
from indy_loco.experiment.phase17_architecture_comparison import data_contract as contract
from indy_loco.experiment.phase17_architecture_comparison.run import exclusive_run

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUTPUT = HERE.parent / "results/size_sweep_v1"
WIDTHS = (96, 128, 256)
TRAINING = SimpleNamespace(init="midsize_transfer", train_scope="all", epochs=20,
                           batch_size=128, patience=3, weight_decay=.025, gradient_clip=1.)


def write(path, value):
    protocol.write_json_atomic(path, value)


def sources():
    files = list(HERE.glob("*.py"))
    root = HERE.parent.parent
    files += list((root / "phase16_parameter_scaling").glob("*.py"))
    files += [root / "phase16_parameter_scaling/protocol_lock.json",
              root / "phase17_architecture_comparison/data_contract.py",
              root / "phase17_architecture_comparison/run.py"]
    return {str(p.relative_to(REPO)): protocol.sha256_file(p) for p in files}


def reports(rows, config):
    summary = {}
    for width in WIDTHS:
        selected = [r for r in rows if r["width"] == width]
        scores = [r["validation"]["r2_mean"] for r in selected]
        summary[str(width)] = {
            "completed_folds": len(scores), "expected_folds": 30,
            "validation_r2_mean": statistics.mean(scores) if scores else None,
            "validation_r2_sample_sd": statistics.stdev(scores) if len(scores) > 1 else None,
            "capacity": config["capacity"][str(width)],
        }
    complete = len(rows) == 90
    result = {"status": "complete" if complete else "partial", "completed_jobs": len(rows),
              "expected_jobs": 90, "test_evaluated": False, "summary": summary,
              "selection": "Rank only after all 30 matched validation folds per width complete",
              "caveat": "Warm-started from same-fold Midsize selected on validation. This is a tuning comparison, not an unbiased final test estimate.",
              "results": rows}
    if complete:
        result["validation_ranking"] = sorted(WIDTHS, key=lambda w: summary[str(w)]["validation_r2_mean"], reverse=True)
    write(OUTPUT / "metrics.json", result)


def run(args):
    contract.verify_protocol_lock()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    sessions = list(contract.session_data.SESSION_BY_NAME)
    architectures = {w: Architecture(w, 3, (1, 2, 4, 8), w, 1) for w in WIDTHS}
    config = {"phase": "phase19_size_sweep_v1", "widths": list(WIDTHS), "sessions": sessions,
              "folds": list(range(1, 6)), "seed": 43, "training": vars(TRAINING),
              "stopping": StopCriterion().metadata(), "learning_rate_gru_head": 3e-4,
              "encoder_lr_scale": .25, "kernels": "native PyTorch/cuDNN, FP32, no TF32",
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "capacity": {str(w): parameter_report(a) for w, a in architectures.items()},
              "source_sha256": sources(), "test_evaluated": False,
              "references": {f"{s}_fold{f}": protocol.sha256_file(contract.reference_path(s, f))
                             for s in sessions for f in range(1, 6)}}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    saved = OUTPUT / "config.json"
    if saved.exists():
        if not args.resume:
            raise FileExistsError("Existing sweep: use --resume to verify and skip completed folds")
        if json.loads(saved.read_text(encoding="utf-8")) != config:
            raise ValueError("Sweep configuration/source/reference changed; refusing resume")
    else:
        write(saved, config)
    if args.validate_only:
        for w, architecture in architectures.items():
            protocol.seed_everything(43, mps=False)
            model = ScaledTCNGRU(architecture).cuda()
            reference = contract.load_reference(sessions[0], 1)
            transfer_midsize_weights(model, reference["model_state"])
            y = model(torch.randn(2, 192, 50, device="cuda"))
            assert y.shape == (2, 50, 2)
            y.square().mean().backward()
            if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise AssertionError(f"Invalid gradients for width{w}")
            print(f"width{w}: GPU forward/backward/transfer passed; {config['capacity'][str(w)]}", flush=True)
            del model, y, reference
        write(OUTPUT / "preflight.json", {"status": "passed", "widths": list(WIDTHS), "signature": signature})
        return
    contract.configure_paths(contract.INDY / "data", contract.default_gui_root(), OUTPUT / ".cache")
    rows = []
    for session in sessions:
        raw = contract.load_session(session, contract.REFERENCE.parent / ".cache/session_inputs")
        for fold in range(1, 6):
            reference = contract.load_reference(session, fold)
            data, evidence = contract.prepare_verified_fold(raw, fold, reference)
            for width in WIDTHS:
                identity = f"w{width}_{session}_fold{fold}"
                result_path = OUTPUT / "fold_results" / f"{identity}.json"
                checkpoint_path = OUTPUT / "checkpoints" / f"{identity}.pt"
                if result_path.exists():
                    row = json.loads(result_path.read_text(encoding="utf-8"))
                    if (row["signature"] != signature or row["preprocessing_evidence"] != evidence
                            or row["checkpoint_sha256"] != protocol.sha256_file(checkpoint_path)):
                        raise ValueError(f"Saved result integrity failure: {identity}")
                    rows.append(row)
                    reports(rows, config)
                    print(f"Verified completed {identity}", flush=True)
                    continue
                if sources() != config["source_sha256"]:
                    raise ValueError("Source changed during sweep")
                # Preserve a checkpoint left by interruption before the result JSON.
                if checkpoint_path.exists():
                    archived = checkpoint_path.with_name(checkpoint_path.stem + f".interrupted_{time.time_ns()}.pt")
                    checkpoint_path.rename(archived)
                write(OUTPUT / "status.json", {"status": "training", "width": width,
                      "session": session, "fold": fold, "completed_jobs": len(rows), "expected_jobs": 90})
                print(f"\nSTART {identity} ({len(rows)}/90 complete)", flush=True)
                protocol.seed_everything(43, mps=False)
                model = ScaledTCNGRU(architectures[width])
                transfer = transfer_midsize_weights(model, reference["model_state"])
                model.cuda()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                best, epoch, history, loss = fast_training.fit_model(
                    model, raw, fold - 1, data, TRAINING, torch.device("cuda"), 3e-4, .25)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                validation = protocol.evaluate_last(model, data.normalized_features, data.velocity,
                    data.validation_bins, data.target_mean, data.target_std, torch.device("cuda"), 128)
                checkpoint = {"model_state": best, "architecture": architectures[width].metadata(),
                              "session": session, "fold": fold, "seed": 43, "signature": signature,
                              "preprocessing_evidence": evidence, "weight_transfer": transfer,
                              "best_epoch": epoch, "test_evaluated": False,
                              "preprocessing": {k: reference[k] for k in (
                                  "selected_channel_indices", "feature_mean", "feature_std", "target_mean",
                                  "target_std", "deployment_policy", "feature_std_floor", "calibration_local_std")}}
                protocol.save_checkpoint_atomic(checkpoint_path, checkpoint)
                row = {"width": width, "session": session, "fold": fold, "signature": signature,
                       "best_epoch": epoch, "best_validation_loss": loss, "validation": validation,
                       "training_seconds": elapsed, "epochs_ran": len(history), "history": history,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "checkpoint": str(checkpoint_path.relative_to(OUTPUT)),
                       "checkpoint_sha256": protocol.sha256_file(checkpoint_path),
                       "preprocessing_evidence": evidence}
                write(result_path, row)
                rows.append(row)
                reports(rows, config)
                print(f"DONE {identity}: val R2={validation['r2_mean']:.6f}, {elapsed:.1f}s", flush=True)
                del model, best, checkpoint
                gc.collect()
                torch.cuda.empty_cache()
            del data, reference
        del raw
    write(OUTPUT / "status.json", {"status": "complete", "completed_jobs": 90, "test_evaluated": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    lock = HERE.parent.parent / "phase17_architecture_comparison/results/.phase17_gpu"
    with exclusive_run(lock):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoints", "fold_results"):
            (OUTPUT / name).mkdir(exist_ok=True)
        try:
            run(args)
        except BaseException as error:
            write(OUTPUT / "last_error.json", {"error": str(error), "traceback": traceback.format_exc()})
            write(OUTPUT / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                           "error": str(error)})
            raise


if __name__ == "__main__":
    main()
