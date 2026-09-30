"""Train/validation-only error diagnosis of the frozen r07 checkpoints.

No optimizer, model edits, test metrics, test predictions, or test-label analysis.
The frozen preparer may verify the original test split as part of its contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[2]
sys.path.insert(0, str(REPO))
SOURCE = PHASE / "results/regularization_v2"
OUTPUT = PHASE / "results/ab_diagnostics_v1"
LAGS = tuple(range(-3, 4))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def axis_metrics(target, prediction):
    target, prediction = (
        np.asarray(target, np.float64),
        np.asarray(prediction, np.float64),
    )
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("Expected paired N x 2 arrays")
    if (
        len(target) < 2
        or not np.isfinite(target).all()
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("Insufficient or nonfinite paired predictions")
    result = []
    for axis in range(2):
        y, p = target[:, axis], prediction[:, axis]
        error = p - y
        variance = float(np.mean((y - y.mean()) ** 2))
        p_variance = float(np.mean((p - p.mean()) ** 2))
        covariance = float(np.mean((p - p.mean()) * (y - y.mean())))
        result.append(
            {
                "axis": "xy"[axis],
                "samples": len(y),
                "bias": float(error.mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "r2": 1 - float(np.mean(error**2)) / variance if variance > 0 else None,
                "correlation_squared": covariance**2 / (variance * p_variance)
                if variance * p_variance > 0
                else None,
                "prediction_target_std_ratio": math.sqrt(p_variance / variance)
                if variance > 0
                else None,
                "target_std": math.sqrt(variance),
            }
        )
    return result


def fit_training_affine(training_target, training_prediction):
    """Independent per-axis least squares; receives no validation labels."""
    y, p = (
        np.asarray(training_target, np.float64),
        np.asarray(training_prediction, np.float64),
    )
    if (
        y.shape != p.shape
        or y.ndim != 2
        or y.shape[1] != 2
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("Invalid training-only affine inputs")
    gains, offsets = [], []
    for axis in range(2):
        centered = p[:, axis] - p[:, axis].mean()
        denominator = float(centered @ centered)
        gain = (
            float(centered @ (y[:, axis] - y[:, axis].mean())) / denominator
            if denominator > 1e-16
            else 0.0
        )
        gains.append(gain)
        offsets.append(float(y[:, axis].mean() - gain * p[:, axis].mean()))
    return np.asarray(gains), np.asarray(offsets)


def reach_ids(bins, bounds, selected_reaches):
    bins = np.asarray(bins, np.int64)
    if len(bins) == 0 or np.any(np.diff(bins) <= 0):
        raise ValueError("Bins must be strictly increasing")
    ids = np.full(len(bins), -1, dtype=np.int64)
    for reach in np.asarray(selected_reaches, np.int64):
        start, stop = bounds[reach]
        inside = (bins >= start) & (bins < stop)
        if np.any(ids[inside] != -1):
            raise ValueError("Overlapping selected reaches")
        ids[inside] = reach
    if np.any(ids < 0):
        raise ValueError("Selected bins fall outside selected reaches")
    return ids


def common_lag_pairs(bins, ids, lags=LAGS):
    """Return one target subset and prediction indices for every lag.

    Each comparison is prediction(t - lag) versus target(t). Thus negative lag
    needs later predictions and can indicate delayed output; it is not causal
    deployment. All target bins are identical across lags, and neither gaps nor
    reach boundaries are crossed.
    """
    bins, ids = np.asarray(bins), np.asarray(ids)
    index = {int(value): position for position, value in enumerate(bins)}
    targets, mapped = [], {lag: [] for lag in lags}
    for position, value in enumerate(bins):
        candidates = {lag: index.get(int(value - lag)) for lag in lags}
        if all(
            other is not None and ids[other] == ids[position]
            for other in candidates.values()
        ):
            targets.append(position)
            for lag, other in candidates.items():
                mapped[lag].append(other)
    return np.asarray(targets, np.int64), {
        lag: np.asarray(values, np.int64) for lag, values in mapped.items()
    }


def motion_values(bins, ids, target):
    speed = np.linalg.norm(target, axis=1)
    acceleration = np.full(len(bins), np.nan)
    adjacent = (np.diff(bins) == 1) & (np.diff(ids) == 0)
    positions = np.flatnonzero(adjacent) + 1
    acceleration[positions] = (
        np.linalg.norm(target[positions] - target[positions - 1], axis=1) / 0.04
    )
    return speed, acceleration


def regime_diagnostics(
    train_bins, train_ids, train_target, val_bins, val_ids, val_target, val_prediction
):
    train_speed, train_acceleration = motion_values(train_bins, train_ids, train_target)
    speed_low, speed_high = np.quantile(train_speed, [0.2, 0.8])
    finite_acceleration = train_acceleration[np.isfinite(train_acceleration)]
    if not len(finite_acceleration):
        raise ValueError("No contiguous training accelerations")
    acc_high = float(np.quantile(finite_acceleration, 0.8))
    speed, acceleration = motion_values(val_bins, val_ids, val_target)
    masks = {
        "speed_low": speed <= speed_low,
        "speed_middle": (speed > speed_low) & (speed < speed_high),
        "speed_high": speed >= speed_high,
        "acceleration_low_middle": np.isfinite(acceleration)
        & (acceleration < acc_high),
        "acceleration_high": np.isfinite(acceleration) & (acceleration >= acc_high),
        "x_positive": val_target[:, 0] > 0,
        "x_negative": val_target[:, 0] < 0,
        "y_positive": val_target[:, 1] > 0,
        "y_negative": val_target[:, 1] < 0,
    }
    return {
        "thresholds_from_training_only": {
            "speed_q20": float(speed_low),
            "speed_q80": float(speed_high),
            "acceleration_q80": acc_high,
        },
        "validation_groups": {
            name: {
                "samples": int(mask.sum()),
                "axes": axis_metrics(val_target[mask], val_prediction[mask])
                if mask.sum() >= 2
                else [],
            }
            for name, mask in masks.items()
        },
    }


def diagnose_arrays(arrays, bounds, train_reaches, validation_reaches):
    train_y, train_p = arrays["train_target"], arrays["train_prediction"]
    val_y, val_p = arrays["validation_target"], arrays["validation_prediction"]
    train_ids = reach_ids(arrays["train_bins"], bounds, train_reaches)
    val_ids = reach_ids(arrays["validation_bins"], bounds, validation_reaches)
    gains, offsets = fit_training_affine(train_y, train_p)
    target_indices, prediction_indices = common_lag_pairs(
        arrays["validation_bins"], val_ids
    )
    if len(target_indices) < 2:
        raise ValueError("No common validation interior for lag analysis")
    lags = [
        {
            "lag_bins": lag,
            "lag_ms": lag * 40,
            "axes": axis_metrics(val_y[target_indices], val_p[prediction_indices[lag]]),
        }
        for lag in LAGS
    ]
    return {
        "train": axis_metrics(train_y, train_p),
        "validation": axis_metrics(val_y, val_p),
        "training_affine": {
            "gain": gains.tolist(),
            "offset": offsets.tolist(),
            "validation": axis_metrics(val_y, val_p * gains + offsets),
        },
        "lag_diagnostic": {
            "common_target_samples": len(target_indices),
            "results": lags,
        },
        "motion_regimes": regime_diagnostics(
            arrays["train_bins"],
            train_ids,
            train_y,
            arrays["validation_bins"],
            val_ids,
            val_y,
            val_p,
        ),
    }


def summarize(folds):
    axes = []
    for axis in range(2):
        rows = [fold["validation"][axis] for fold in folds]
        affine = [fold["training_affine"]["validation"][axis] for fold in folds]
        axes.append(
            {
                "axis": "xy"[axis],
                "folds": len(rows),
                **{
                    key: statistics.mean(row[key] for row in rows)
                    for key in (
                        "r2",
                        "rmse",
                        "bias",
                        "correlation_squared",
                        "prediction_target_std_ratio",
                    )
                },
                "affine_validation_r2": statistics.mean(row["r2"] for row in affine),
                "affine_delta_r2": statistics.mean(
                    b["r2"] - a["r2"] for a, b in zip(rows, affine, strict=True)
                ),
                "affine_improved_folds": sum(
                    b["r2"] > a["r2"] for a, b in zip(rows, affine, strict=True)
                ),
                "train_fit_gain_mean": statistics.mean(
                    fold["training_affine"]["gain"][axis] for fold in folds
                ),
            }
        )
    lag_rows = []
    for lag in LAGS:
        lag_rows.append(
            {
                "lag_bins": lag,
                "axes": [
                    {
                        "axis": "xy"[axis],
                        "r2_mean": statistics.mean(
                            next(
                                row
                                for row in fold["lag_diagnostic"]["results"]
                                if row["lag_bins"] == lag
                            )["axes"][axis]["r2"]
                            for fold in folds
                        ),
                    }
                    for axis in range(2)
                ],
            }
        )
    return {"axes": axes, "lag_curve": lag_rows}


def figure(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    sessions = list(summary["by_session"])
    positions = np.arange(len(sessions))
    for index, color in enumerate(("#1768AC", "#D65F28")):
        axes[0].plot(
            positions,
            [summary["by_session"][s]["axes"][index]["r2"] for s in sessions],
            "o-",
            color=color,
            label="xy"[index],
        )
        axes[1].plot(
            [r["lag_bins"] * 40 for r in summary["overall"]["lag_curve"]],
            [r["axes"][index]["r2_mean"] for r in summary["overall"]["lag_curve"]],
            "o-",
            color=color,
            label="xy"[index],
        )
        row = summary["overall"]["axes"][index]
        axes[2].bar(index - 0.17, row["r2"], 0.34, color=color, alpha=0.45)
        axes[2].bar(index + 0.17, row["affine_validation_r2"], 0.34, color=color)
    axes[0].set_xticks(
        positions,
        [s.replace("indy_", "I ").replace("loco_", "L ") for s in sessions],
        rotation=42,
        ha="right",
        fontsize=8,
    )
    axes[0].set_title("Validation R² by session")
    axes[0].legend(frameon=False)
    axes[1].set_title("Timing diagnostic: same target bins")
    axes[1].set_xlabel("lag (ms): prediction(t − lag) vs target(t)")
    axes[1].axvline(0, color="gray", lw=0.8)
    axes[2].set_title("Train-fit affine correction → validation")
    axes[2].set_xticks([0, 1], ["x", "y"])
    axes[2].text(
        0.5,
        0.04,
        "Light: original · Dark: train-fit affine",
        ha="center",
        transform=axes[2].transAxes,
        fontsize=8,
    )
    for ax in axes:
        ax.set_ylabel("Fold-macro R²")
        ax.grid(axis="y", alpha=0.16)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Frozen r07 error diagnosis — training + validation only",
        fontsize=16,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout()
    path = output / "validation_diagnostics.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path.name


def write_report(summary, output):
    overall = summary["overall"]
    lines = [
        "# Validation error diagnosis / 验证误差诊断",
        "",
        "All 30 frozen r07 checkpoints verified; train/validation predictions only. No test metrics, predictions or test-label analysis. No calibration or timing change is deployed.",
        "全部 30 个 r07 checkpoint 已核验；只分析 train/validation，未分析 test 指标、预测或标签，也未部署校正。",
        "",
        "| Axis | Validation R² | Corr² | Pred/target SD | Train-fit affine ΔR² | Improved folds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall["axes"]:
        lines.append(
            f"| {row['axis']} | {row['r2']:.4f} | {row['correlation_squared']:.4f} | {row['prediction_target_std_ratio']:.3f} | {row['affine_delta_r2']:+.4f} | {row['affine_improved_folds']}/30 |"
        )
    lines += [
        "",
        "Affine slopes/intercepts were fitted on training predictions and labels only and then scored on validation. Correlation squared is descriptive; it does not authorize fitting a calibration to the validation targets.",
        "仿射 gain/offset 仅由训练集拟合，再在 validation 上评分；相关系数平方不是直接用 validation 拟合校正的许可。",
        "",
        "| Lag | x R² | y R² |",
        "|---|---:|---:|",
    ]
    for row in overall["lag_curve"]:
        lines.append(
            f"| {row['lag_bins'] * 40:+d} ms | {row['axes'][0]['r2_mean']:.4f} | {row['axes'][1]['r2_mean']:.4f} |"
        )
    lines += [
        "",
        "Lag convention: prediction(t − lag) versus target(t). Negative lag uses a later prediction to match the current target and can indicate a lagging decoder; it cannot be deployed causally. All seven lags use exactly the same interior target bins, within the same contiguous validation reach. These subset scores need not equal full-validation scores.",
        "lag 定义：prediction(t − lag) 对 target(t)。负 lag 会使用更晚的预测，因此只是延迟诊断、不可因果部署。七个 lag 使用相同 target 样本，且严格限制在同一 validation reach 连续片段内。",
        "",
        "Speed q20/q80 and acceleration q80 thresholds are learned separately from each training fold. Acceleration uses only adjacent bins in the same selected reach. Group RMSE/R² and sample counts are saved in summary.json; low-variance subgroups can have unstable R², so use RMSE and counts alongside it.",
        "速度和加速度分组阈值仅来自训练集；加速度不跨 split、gap 或 reach。分组结果见 summary.json，低方差分组的 R² 需结合 RMSE 和样本量解释。",
        "",
        f"Diagnostic gate: **{summary['diagnostic_gate']}**. Predictions reproduce stored validation R² within 5e-6; no preprocessing changes were made. Lag or scale differences are hypotheses, not a confirmed source-data bug.",
        "诊断只提供后续模型比较的依据；不会自动更改数据对齐、归一化或标签。",
        "",
        f"![Validation diagnostics]({summary['figure']})",
        "",
        "All aggregates weight folds equally. Cross-validation folds overlap in training and are not independent replicates; no significance claim is made.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output == SOURCE.resolve() or SOURCE.resolve() in output.parents:
        raise ValueError("Output must not overwrite frozen results")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from indy_loco.experiment.phase18_large_mingru.tuning import train as frozen

    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    config = read_json(SOURCE / "config.json")
    signature = frozen.digest(config)
    hashes = {str(SOURCE / "config.json"): sha256(SOURCE / "config.json")}
    for relative, expected in config["code_sha256"].items():
        actual = hashlib.sha256(
            (REPO / relative).read_text(encoding="utf-8").encode()
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen code changed: {relative}")
    frozen.contract.verify_protocol_lock()
    frozen.contract.configure_paths(
        frozen.contract.INDY / "data",
        frozen.contract.default_gui_root(),
        output / ".cache",
    )
    device = frozen.protocol.select_device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    folds = []
    with ExitStack() as stack:
        stack.enter_context(
            frozen.original.exclusive_run(frozen.RESULTS / ".phase17_gpu")
        )
        stack.enter_context(frozen.original.exclusive_run(output))
        for session in frozen.SESSIONS:
            for fold in range(1, 6):
                label = f"{session}/fold{fold}"
                print(f"DIAGNOSE {label}", flush=True)
                data, evidence = frozen.prepare(session, fold)
                checkpoint = (
                    SOURCE
                    / "runs/mingru_large/r07/seed43/checkpoints"
                    / f"{session}_fold{fold}.pt"
                )
                receipt_path = checkpoint.with_suffix(".validation.json")
                receipt = read_json(receipt_path)
                saved = frozen.original.load_training_checkpoint(checkpoint)
                frozen.check_saved(
                    saved,
                    frozen.identity(
                        "mingru_large", "r07", 43, session, fold, signature
                    ),
                    evidence,
                )
                checksum = sha256(checkpoint)
                if (
                    checksum != receipt["checkpoint_sha256"]
                    or saved["validation"] != receipt["validation"]
                ):
                    raise ValueError("Checkpoint/receipt mismatch")
                for key in (
                    "target_mean",
                    "target_std",
                    "calibration_mean",
                    "calibration_effective_std",
                    "channels",
                ):
                    np.testing.assert_array_equal(
                        saved["scalers"][key], getattr(data, key)
                    )
                for key in (
                    "train_bins",
                    "validation_bins",
                    "train_reaches",
                    "validation_reaches",
                ):
                    np.testing.assert_array_equal(
                        saved["split_indices"][key], getattr(data, key)
                    )
                if np.intersect1d(data.train_bins, data.validation_bins).size:
                    raise ValueError("Train/validation overlap")
                model = (
                    frozen.create_model("mingru_large", config["trials"]["r07"])
                    .to(device)
                    .eval()
                )
                model.load_state_dict(saved["model_state"])
                arrays = {}
                for split in ("train", "validation"):
                    bins = getattr(data, split + "_bins")
                    arrays[split + "_bins"] = bins.copy()
                    arrays[split + "_target"] = data.velocity[bins].copy()
                    arrays[split + "_prediction"] = frozen.protocol.predict_last(
                        model,
                        data.normalized_features,
                        bins,
                        data.target_mean,
                        data.target_std,
                        device,
                        256,
                    )
                measured = frozen.protocol.metrics(
                    arrays["validation_target"], arrays["validation_prediction"]
                )
                for key in ("r2_x", "r2_y", "r2_mean"):
                    if abs(measured[key] - saved["validation"][key]) > 5e-6:
                        raise ValueError(
                            f"Validation reproducibility failed: {label}/{key}"
                        )
                artifact = output / "predictions" / f"{session}_fold{fold}.npz"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    artifact, **arrays, checkpoint_sha256=np.asarray(checksum)
                )
                result = {
                    "session": session,
                    "fold": fold,
                    "checkpoint_sha256": checksum,
                    "prediction_file": artifact.relative_to(output).as_posix(),
                    "prediction_sha256": sha256(artifact),
                    **diagnose_arrays(
                        arrays, data.bounds, data.train_reaches, data.validation_reaches
                    ),
                }
                folds.append(result)
                write_json(output / "folds" / f"{session}_fold{fold}.json", result)
                hashes[str(checkpoint)], hashes[str(receipt_path)] = (
                    checksum,
                    sha256(receipt_path),
                )
                print(
                    f"VERIFIED {len(folds)}/30 val_R2={measured['r2_mean']:.6f}",
                    flush=True,
                )
                del model, saved, arrays
        summary = {
            "status": "complete",
            "verified_folds": len(folds),
            "test_analysis_performed": False,
            "calibration_fit_split": "train_only",
            "diagnostic_gate": "pass_reproducibility_and_split_integrity",
            "input_hashes": hashes,
            "source_input_manifest": config["inputs"],
            "overall": summarize(folds),
            "by_session": {
                s: summarize([f for f in folds if f["session"] == s])
                for s in frozen.SESSIONS
            },
            "folds": folds,
        }
        if len(folds) != 30:
            raise ValueError("Incomplete diagnostics")
        summary["figure"] = figure(summary, output)
        write_json(output / "summary.json", summary)
        rows = [
            {"session": fold["session"], "fold": fold["fold"], **axis}
            for fold in folds
            for axis in fold["validation"]
        ]
        with (output / "validation_axes.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_report(summary, output)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "verified_folds": 30,
                    "overall": summary["overall"],
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
