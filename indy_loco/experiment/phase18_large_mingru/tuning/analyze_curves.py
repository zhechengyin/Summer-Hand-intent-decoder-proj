"""Audit Phase18 learning curves using validation evidence only.

Reads config, validation-based screening selections, epoch histories, validation
receipts and checkpoint hashes. Never reads final test metrics or predictions.
Writes derived files exclusively to a separate analysis output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PHASE / "results" / "large_mingru_v1"
DEFAULT_OUTPUT = PHASE / "results" / "regularization_v2_analysis"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def mean(rows, key):
    return statistics.mean(row[key] for row in rows)


def aggregate(rows):
    return {
        "fits": len(rows),
        "stopped_before_maximum": sum(row["early_stopped"] for row in rows),
        "reached_maximum": sum(not row["early_stopped"] for row in rows),
        "best_epoch_mean": mean(rows, "best_epoch"),
        "best_epoch_median": statistics.median(row["best_epoch"] for row in rows),
        "best_epoch_min": min(row["best_epoch"] for row in rows),
        "best_epoch_max": max(row["best_epoch"] for row in rows),
        "last_epoch_median": statistics.median(row["last_epoch"] for row in rows),
        "best_at_last_epoch": sum(
            row["best_epoch"] == row["last_epoch"] for row in rows
        ),
        "best_epoch_at_most_30": sum(row["best_epoch"] <= 30 for row in rows),
        "training_falls_validation_rises_after_best": sum(
            row["train_last"] < row["train_at_best"]
            and row["validation_last"] > row["validation_best"]
            for row in rows
        ),
        "training_optimization_loss_at_best_mean": mean(rows, "train_at_best"),
        "training_optimization_loss_at_last_mean": mean(rows, "train_last"),
        "validation_normalized_mse_best_mean": mean(rows, "validation_best"),
        "validation_normalized_mse_last_mean": mean(rows, "validation_last"),
        "validation_mse_increase_after_best_mean": mean(rows, "validation_increase"),
        "validation_mse_increase_after_best_relative_percent_mean": mean(
            rows, "validation_increase_percent"
        ),
        "descriptive_val_minus_train_at_best_mean": mean(rows, "gap_at_best"),
        "descriptive_val_minus_train_at_last_mean": mean(rows, "gap_at_last"),
        "validation_r2_at_mse_selected_checkpoint_mean": mean(rows, "validation_r2"),
    }


def load_evidence(source):
    config = read_json(source / "config.json")
    promotion = read_json(source / "screen_selection.json")["promoted"]
    maximum = config["fixed"]["epochs"]
    patience = config["fixed"]["patience"]
    expected = {
        (model, trial, 43, session, fold)
        for model in config["models"]
        for trial in config["trials"]
        for session in config["sessions"]
        for fold in config["folds"]
        if fold in config["screen_folds"] or trial in promotion[model]
    }
    rows, histories, identities = [], {}, set()
    sources = {
        "config.json": sha256(source / "config.json"),
        "screen_selection.json": sha256(source / "screen_selection.json"),
    }
    for epoch_path in sorted(source.glob("runs/*/*/seed*/epochs/*.json")):
        checkpoint = (
            epoch_path.parent.parent / "checkpoints" / (epoch_path.stem + ".pt")
        )
        receipt_path = checkpoint.with_suffix(".validation.json")
        receipt = read_json(receipt_path)
        history = read_json(epoch_path)
        identity = tuple(
            receipt[key] for key in ("model", "trial", "seed", "session", "fold")
        )
        if identity in identities or identity not in expected:
            raise ValueError(f"Unexpected/duplicate fit: {identity}")
        identities.add(identity)
        if receipt.get("test_evaluated_during_training") is not False:
            raise ValueError(
                f"Missing training/test isolation declaration: {receipt_path}"
            )
        checksum = sha256(checkpoint)
        if checksum != receipt["checkpoint_sha256"]:
            raise ValueError(
                f"Checkpoint differs from validation receipt: {checkpoint}"
            )
        if checksum != read_json(checkpoint.with_suffix(".sha256.json"))["sha256"]:
            raise ValueError(f"Checkpoint sidecar hash mismatch: {checkpoint}")
        if not history or len(history) > maximum:
            raise ValueError(f"Invalid epoch count: {epoch_path}")
        if [epoch["epoch"] for epoch in history] != list(range(1, len(history) + 1)):
            raise ValueError(f"Non-contiguous epoch history: {epoch_path}")
        if not all(
            math.isfinite(float(value))
            for epoch in history
            for value in (
                epoch["optimization_loss"],
                epoch["validation"]["normalized_loss"],
                epoch["validation"]["r2_mean"],
            )
        ):
            raise ValueError(f"Nonfinite learning curve: {epoch_path}")
        best_index = min(
            range(len(history)),
            key=lambda index: history[index]["validation"]["normalized_loss"],
        )
        best, last = history[best_index], history[-1]
        if (
            receipt["best_epoch"] != best_index + 1
            or receipt["validation"] != best["validation"]
        ):
            raise ValueError(
                f"Selected checkpoint does not match epoch evidence: {epoch_path}"
            )
        early = len(history) < maximum
        if early and len(history) - receipt["best_epoch"] != patience:
            raise ValueError(f"Early stop inconsistent with patience: {epoch_path}")
        best_loss = best["validation"]["normalized_loss"]
        last_loss = last["validation"]["normalized_loss"]
        row = {
            **{
                key: receipt[key]
                for key in ("model", "trial", "seed", "session", "fold")
            },
            "last_epoch": len(history),
            "best_epoch": receipt["best_epoch"],
            "early_stopped": early,
            "epochs_after_best": len(history) - receipt["best_epoch"],
            "train_first": history[0]["optimization_loss"],
            "train_at_best": best["optimization_loss"],
            "train_last": last["optimization_loss"],
            "validation_first": history[0]["validation"]["normalized_loss"],
            "validation_best": best_loss,
            "validation_last": last_loss,
            "validation_increase": last_loss - best_loss,
            "validation_increase_percent": 100 * (last_loss / best_loss - 1),
            "gap_at_best": best_loss - best["optimization_loss"],
            "gap_at_last": last_loss - last["optimization_loss"],
            "validation_r2": receipt["validation"]["r2_mean"],
            "checkpoint_sha256": checksum,
            "epochs_file": epoch_path.relative_to(source).as_posix(),
            "receipt_file": receipt_path.relative_to(source).as_posix(),
        }
        rows.append(row)
        histories[identity] = history
        sources[row["epochs_file"]] = sha256(epoch_path)
        sources[row["receipt_file"]] = sha256(receipt_path)
    if identities != expected or len(rows) != config["fit_budget"]["total"]:
        raise ValueError(
            f"Incomplete evidence: {len(rows)} fits, expected {len(expected)}"
        )
    return config, rows, histories, sources


def plot_t00(config, rows, histories, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(3, 2, figsize=(13.4, 11.3), sharex=True)
    train_color, val_color = "#1768AC", "#D65F28"
    for ax, session in zip(axes.flat, config["sessions"], strict=True):
        selected = sorted(
            [
                row
                for row in rows
                if row["trial"] == "t00" and row["session"] == session
            ],
            key=lambda row: row["fold"],
        )
        for row in selected:
            key = tuple(
                row[field] for field in ("model", "trial", "seed", "session", "fold")
            )
            history = histories[key]
            epochs = [item["epoch"] for item in history]
            strong = row["fold"] == 1
            kwargs = {
                "alpha": 0.95 if strong else 0.32,
                "linewidth": 2.0 if strong else 1.1,
            }
            ax.plot(
                epochs,
                [item["optimization_loss"] for item in history],
                color=train_color,
                **kwargs,
            )
            ax.plot(
                epochs,
                [item["validation"]["normalized_loss"] for item in history],
                color=val_color,
                **kwargs,
            )
            ax.scatter(
                row["best_epoch"],
                row["validation_best"],
                s=29 if strong else 19,
                color=val_color,
                edgecolors="white",
                linewidths=0.55,
                zorder=5,
            )
        ax.set_title(
            session.replace("_", " "), loc="left", fontweight="bold", fontsize=11
        )
        ax.text(
            0.97,
            0.95,
            "Best epochs: " + ", ".join(str(row["best_epoch"]) for row in selected),
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8.6,
        )
        ax.set_xlim(1, config["fixed"]["epochs"])
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Normalized MSE")
        ax.grid(alpha=0.16)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    handles = [
        Line2D(
            [0],
            [0],
            color=train_color,
            lw=2,
            label="Training optimization loss (dropout ON)",
        ),
        Line2D(
            [0],
            [0],
            color=val_color,
            lw=2,
            label="Validation loss (eval / dropout OFF)",
        ),
        Line2D(
            [0], [0], color=val_color, marker="o", lw=0, label="Minimum validation MSE"
        ),
    ]
    fig.suptitle(
        "Phase18 t00 learning curves — all 30 folds",
        fontsize=17,
        fontweight="bold",
        x=0.06,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.06,
        0.954,
        "Each session shows all 5 folds; fold 1 is bold. Curves end at their recorded stopping epoch.",
        fontsize=10,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.031),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.06,
        0.012,
        "Training uses changing weights and active dropout; absolute train–validation gaps are descriptive, not a matched generalization estimate.",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0.025, 0.082, 0.995, 0.935), h_pad=2.0, w_pad=2.0)
    destination = output / "t00_learning_curves_by_session.png"
    fig.savefig(destination, dpi=190, facecolor="white")
    plt.close(fig)
    return destination.name


def report(summary):
    overall, trials = summary["overall"], summary["by_trial"]
    lines = [
        "# Phase18 learning curves / 学习曲线分析",
        "",
        "**Regularization is the next priority; these curves do not justify a blanket increase in training length.**",
        "**下一步优先测试正则化；目前曲线不支持统一延长训练。**",
        "",
        f"Audited all {overall['fits']} fits using epoch histories and validation receipts. All checkpoint hashes and selected-epoch scores match. No test metrics or predictions were read.",
        f"已核对全部 {overall['fits']} 次训练的逐 epoch 日志、validation 收据和 checkpoint 哈希；未读取 test 指标或预测。",
        "",
        f"- {overall['stopped_before_maximum']}/{overall['fits']} stopped before 60 epochs under patience 15; best epoch median {overall['best_epoch_median']:g}, range {overall['best_epoch_min']}–{overall['best_epoch_max']}.",
        f"- {overall['best_epoch_at_most_30']}/{overall['fits']} selected their best checkpoint by epoch 30; none selected the last recorded epoch.",
        f"- In {overall['training_falls_validation_rises_after_best']}/{overall['fits']} fits, training optimization loss was lower at the end than at the selected epoch, while validation MSE was higher. This endpoint pattern is consistent with overfitting; it does not mean every intervening epoch worsened monotonically.",
        "- 79/84 次训练提前停止；所有训练的最终 optimization loss 都比最佳 checkpoint 所在 epoch 更低，但最终 validation MSE 更高。这支持先研究正则化，不代表验证误差逐 epoch 单调上升。",
        "- The selected checkpoint is, by definition, the minimum validation MSE, and early stopping waits for non-improvement. Best-to-last validation degradation is therefore partly built into selection; it is not an independent statistical test of overfitting.",
        "- 最佳 checkpoint 本来就是 validation MSE 的最小值，early stopping 又会等待不再改善；因此最佳到末尾的差值部分来自选择规则，不能独立证明过拟合。",
        "",
        "| Configuration / 配置 | Fits / 次数 | Median best epoch / 最佳 epoch 中位数 | Mean validation MSE: best → last | Mean training loss: selected epoch → last |",
        "|---|---:|---:|---:|---:|",
    ]
    for trial, values in trials.items():
        lines.append(
            f"| {trial} | {values['fits']} | {values['best_epoch_median']:g} | "
            f"{values['validation_normalized_mse_best_mean']:.4f} → {values['validation_normalized_mse_last_mean']:.4f} | "
            f"{values['training_optimization_loss_at_best_mean']:.4f} → {values['training_optimization_loss_at_last_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "t00 and t05 cover all 30 folds; other configurations cover the six screening folds only. Do not compare those unequal-scope means as architecture/configuration rankings.",
            "t00/t05 覆盖完整 30 折，其他配置只覆盖六个初筛折；不能把不同覆盖范围的平均值直接用于排名。",
            "",
            "For the matched 30-fold comparison, t00 validation R² is "
            f"{trials['t00']['validation_r2_at_mse_selected_checkpoint_mean']:.4f}; t05 is "
            f"{trials['t05']['validation_r2_at_mse_selected_checkpoint_mean']:.4f}. "
            "The stronger global dropout/weight-decay combination in t05 did not improve the aggregate over t00. This motivates targeted head dropout as a controlled test, rather than assuming more regularization always helps.",
            "完整 30 折上，更强全局 dropout 与 weight decay 的 t05 未超过 t00。下一步应隔离测试 head dropout，而不是认为正则化越强越好。",
            "",
            "The readout contains 888,578 of 1,211,906 parameters (73.3%), and the frozen model applies dropout to channels and after the stem, with no dropout between readout layers. This is a rationale for trying head regularization, not proof that the head is the accuracy bottleneck.",
            "readout 占参数量 73.3%，原模型在输入和 stem 后使用 dropout，head 中没有 dropout；这提供了测试 head 正则化的理由，但不能证明 head 就是精度瓶颈。",
            "",
            "The five runs reaching 60 epochs selected epochs 46, 47, 46, 53 and 52. Some slow improvement may remain possible after a different schedule or under stronger regularization. Keep the current 60/15 budget for the first controlled comparison, then reconsider length only if the new curves end while validation still improves.",
            "跑满 60 epoch 的五次训练分别在 46、47、46、53、52 选中最佳 checkpoint。更换学习率调度或正则化后可能需要更久；先保持 60/15 的预算，若新曲线末端仍在改善，再单独讨论延长。",
            "",
            "**Loss interpretation:** training optimization loss averages minibatches with changing weights and dropout enabled; validation uses the end-of-epoch model in eval mode on different samples. The CSV includes `gap_at_best` and `gap_at_last` only as descriptive values. They are not a clean train/eval generalization gap. Matched eval-mode training loss was not logged and cannot be reconstructed for every epoch from the best checkpoint alone.",
            "**Loss 解读：** training loss 来自启用 dropout、权重持续更新的 minibatch；validation 使用 epoch 结束后的 eval 模型且样本不同。CSV 中的 gap 仅供描述，不能作为严格的泛化差距；历史日志没有保存每个 epoch 的 eval-mode training loss。",
            "",
            f"![All t00 learning curves]({summary['figure']})",
            "",
            "Files: `per_fit.csv` contains all 84 fits; `aggregate.json` includes per-trial, per-session, matched-screen summaries, source hashes, and the five maximum-budget runs.",
            "报告仅提出下一轮验证假设；不会读取 test 来选参数，也不修改原模型、历史结果或数据划分。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Analysis output must be separate from frozen source results")
    config, rows, histories, sources = load_evidence(source)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": str(source),
        "test_metrics_consulted": False,
        "verified_checkpoint_count": len(rows),
        "overall": aggregate(rows),
        "by_trial": {
            trial: aggregate([row for row in rows if row["trial"] == trial])
            for trial in config["trials"]
        },
        "matched_six_screening_folds_by_trial": {
            trial: aggregate(
                [
                    row
                    for row in rows
                    if row["trial"] == trial and row["fold"] in config["screen_folds"]
                ]
            )
            for trial in config["trials"]
        },
        "t00_by_session": {
            session: aggregate(
                [
                    row
                    for row in rows
                    if row["trial"] == "t00" and row["session"] == session
                ]
            )
            for session in config["sessions"]
        },
        "maximum_budget_runs": [row for row in rows if not row["early_stopped"]],
        "scope_note": "84 fits combine six screening configurations and two full-fold confirmations; per-trial scopes differ.",
        "loss_gap_note": "Training dropout-on, evolving minibatch weights; validation dropout-off, epoch-end model. Gaps are descriptive only.",
        "source_sha256": sources,
    }
    summary["figure"] = plot_t00(config, rows, histories, output)
    with (output / "per_fit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "aggregate.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(report(summary), encoding="utf-8")
    print(json.dumps({"output": str(output), "overall": summary["overall"]}, indent=2))


if __name__ == "__main__":
    main()
