"""Combine saved Phase-17 reports without running models or evaluating test again."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ("mingru", "mamba2", "transformer", "midsize_control")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_metrics(directory, allow_partial):
    payload = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if not allow_partial and not payload["full_30fold"]:
        raise ValueError(f"Incomplete 30-fold run: {directory}")
    seen = set()
    for row in payload["results"]:
        key = (row["session"], row["fold"])
        if key in seen:
            raise ValueError(f"Duplicate fold: {directory}/{key}")
        seen.add(key)
        for field in ("checkpoint", "predictions"):
            if sha256(directory / row[field]) != row[f"{field}_sha256"]:
                raise ValueError(f"Modified {field}: {directory}/{key}")
    if payload["full_30fold"] and len(seen) != 30:
        raise ValueError("Invalid full_30fold flag")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=HERE / "results")
    parser.add_argument(
        "--run",
        action="append",
        type=Path,
        help="Explicit run directories for custom output names",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    directories = args.run or [
        args.results_root / name
        for name in NAMES
        if (args.results_root / name / "metrics.json").is_file()
    ]
    if not directories:
        parser.error("No saved metrics.json found. Train the models first.")
    runs = {}
    for directory in directories:
        payload = checked_metrics(directory, args.allow_partial)
        name = payload["config"]["model"]
        if name in runs:
            raise ValueError(f"Multiple runs for {name}; do not select by test results")
        runs[name] = payload
    if not args.allow_partial and any(name not in runs for name in NAMES[:3]):
        raise ValueError(
            "All three candidate runs are required (or use --allow-partial)"
        )
    first = next(iter(runs.values()))
    shared_keys = (
        "training",
        "sessions",
        "folds",
        "device",
        "threads",
        "inputs",
        "references",
        "environment",
        "code_sha256",
    )
    first_rows = {(r["session"], r["fold"]): r for r in first["results"]}
    for payload in runs.values():
        for key in shared_keys:
            if payload["config"][key] != first["config"][key]:
                raise ValueError(
                    f"Runs differ in {key}; review comparability before combining"
                )
        for row in payload["results"]:
            other = first_rows.get((row["session"], row["fold"]))
            if (
                other
                and row["preprocessing_evidence"] != other["preprocessing_evidence"]
            ):
                raise ValueError("Runs differ in preprocessed arrays or masks")
    control = {
        (r["session"], r["fold"]): r["test"]["r2_mean"]
        for r in runs.get("midsize_control", {}).get("results", [])
    }
    overview, sessions = [], []
    for name, payload in runs.items():
        overall = next(
            row for row in payload["summary"] if row["group"] == "overall_fold_macro"
        )
        paired = [
            row["test"]["r2_mean"] - control[(row["session"], row["fold"])]
            for row in payload["results"]
            if (row["session"], row["fold"]) in control
        ]
        overview.append(
            {
                "model": name,
                "status": payload["status"],
                "full_30fold": payload["full_30fold"],
                **overall,
                "parameters": payload["config"]["capacity"]["parameters"],
                "fp32_weight_kib": payload["config"]["capacity"]["fp32_weight_kib"],
                "delta_vs_scratch_control": statistics.mean(paired) if paired else None,
                "paired_control_folds": len(paired),
            }
        )
        for row in payload["summary"]:
            sessions.append({"model": name, **row})
    output = args.results_root / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("models.csv", overview), ("sessions.csv", sessions)):
        with (output / filename).open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    note = "Historical Midsize: 0.7411375801 +/- 0.0655976930 (30 folds), warm-started; candidates/control are scratch. Within-session folds are correlated. Capacity is FP32 weights, not peak RAM."
    (output / "comparison.json").write_text(
        json.dumps({"models": overview, "sessions": sessions, "note": note}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Phase 17 saved-result comparison",
        "",
        note,
        "",
        "| Model | Folds | Test R2 mean +/- SD | Delta vs historical | Parameters | FP32 KiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overview:
        sd = f"{row['r2_std']:.4f}" if row["r2_std"] is not None else "undefined"
        lines.append(
            f"| {row['model']} | {row['folds']} | {row['r2_mean']:.4f} +/- {sd} | {row['delta_vs_historical_mean']:+.4f} | {row['parameters']:,} | {row['fp32_weight_kib']:.2f} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved comparison: {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
