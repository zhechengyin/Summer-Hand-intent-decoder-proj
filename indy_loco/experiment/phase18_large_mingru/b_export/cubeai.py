"""Run bounded CPU STM32CubeAI conversion stages for an already verified ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUTPUT = HERE.parent / "results/b_export_v1"
CLI = (
    Path.home()
    / "STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Utilities/windows/stedgeai.exe"
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("analyze", "generate", "validate"))
    parser.add_argument("--no-onnx-optimizer", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    verification = json.loads(
        (OUTPUT / "export_verification.json").read_text(encoding="utf-8")
    )
    model = OUTPUT / "mingru_b_fp32_sequential.onnx"
    if sha256(model) != verification["artifacts"][model.name]:
        raise ValueError("Verified ONNX changed")
    if not all(
        item["passed"]
        for split in verification["comparisons"].values()
        for item in split.values()
    ):
        raise ValueError("ONNX parity did not pass")
    label = f"cubeai_{args.stage}{args.tag}"
    directory = OUTPUT / label
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(CLI),
        args.stage,
        "--target",
        "stm32",
        "--type",
        "onnx",
        "--model",
        str(model),
        "--name",
        "mingru_b_fp32",
        "--compression",
        "none",
        "--workspace",
        str(directory / "workspace"),
        "--output",
        str(directory / "output"),
        "--quiet",
    ]
    if args.no_onnx_optimizer:
        command.append("--no-onnx-optimizer")
    if args.stage == "generate":
        command += ["--binary", "--dll", "--split-weights"]
    elif args.stage == "validate":
        command += [
            "--mode",
            "host",
            "--valinput",
            str(OUTPUT / "host_validation_inputs.npy"),
            "--valoutput",
            str(OUTPUT / "host_validation_outputs.npy"),
            "--save-csv",
            "--no-exec-model",
        ]
    started = time.perf_counter()
    record = {
        "command": command,
        "graph_sha256": sha256(model),
        "timeout_seconds": args.timeout,
    }
    with (OUTPUT / f"{label}.log").open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(
                command,
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
            record.update(
                status="complete" if result.returncode == 0 else "failed",
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            record.update(status="timed_out", exit_code=None)
    record["elapsed_seconds"] = time.perf_counter() - started
    record["log"] = str(OUTPUT / f"{label}.log")
    record["artifacts"] = {
        str(p.relative_to(directory)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in directory.rglob("*")
        if p.is_file()
    }
    (OUTPUT / f"{label}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in record.items() if key != "artifacts"},
            indent=2,
        ),
        flush=True,
    )
    print(
        (OUTPUT / f"{label}.log").read_text(encoding="utf-8", errors="replace")[
            -12000:
        ],
        flush=True,
    )
    return 0 if record["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
