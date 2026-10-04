"""Bounded STM32Cube.AI CLI invocation with exact command/log/artifact receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUTPUT = HERE.parent / "results/b2_cubeai_export_v1"
CLI = Path.home() / "STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Utilities/windows/stedgeai.exe"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stage", choices=["analyze", "generate", "validate"], default="generate")
    parser.add_argument("--compression", default="none", choices=["none", "lossless"])
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--no-onnx-optimizer", action="store_true")
    parser.add_argument("--dll", action="store_true")
    parser.add_argument("--external-inputs", action="store_true")
    parser.add_argument("--valinput", type=Path, nargs="+")
    parser.add_argument("--valoutput", type=Path)
    args = parser.parse_args()
    directory = OUTPUT / args.tag
    directory.mkdir(parents=True, exist_ok=False)
    model = args.model.resolve()
    with model.open("rb") as source:
        model_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    command = [str(CLI), args.stage, "--target", "stm32", "--type", "onnx", "--model", str(model),
               "--name", "b2_mingru", "--compression", args.compression,
               "--workspace", str(directory / "workspace"), "--output", str(directory / "output")]
    if args.no_onnx_optimizer:
        command += ["--no-onnx-optimizer"]
    if args.external_inputs:
        command += ["--no-inputs-allocation"]
    command += ["--quiet"]
    if args.stage == "generate":
        command += ["--split-weights"]
        if args.dll:
            command += ["--dll"]
    if args.stage == "validate":
        inputs = args.valinput or [OUTPUT / "host_validation_inputs.npy"]
        output = args.valoutput or OUTPUT / "host_validation_outputs.npy"
        command += ["--mode", "host", "--valinput", *[str(p.resolve()) for p in inputs],
                    "--valoutput", str(output.resolve()), "--no-exec-model"]
    start = time.monotonic()
    peak, reason = 0, None
    with (directory / "command.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        tree = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                members = [tree] + tree.children(recursive=True)
                total = sum(getattr(p.memory_info(), "private", p.memory_info().rss) for p in members if p.is_running())
                peak = max(total, peak)
                if total > args.memory_gib * 1024**3:
                    reason = "memory_limit"
                elif time.monotonic() - start > args.timeout:
                    reason = "timeout"
                if reason:
                    for p in reversed(members):
                        try:
                            p.kill()
                        except psutil.NoSuchProcess:
                            pass
                    break
            except psutil.NoSuchProcess:
                pass
            time.sleep(.25)
        process.wait(timeout=20)
    receipt = {"command": command, "model_sha256": model_sha256, "pid": process.pid, "status": reason or ("complete" if process.returncode == 0 else "failed"),
               "exit_code": process.returncode, "elapsed_seconds": time.monotonic() - start, "peak_private_bytes": peak,
               "limit_gib": args.memory_gib, "log": str(directory / "command.log"),
               "artifacts": {str(p.relative_to(directory)): p.stat().st_size for p in directory.rglob("*") if p.is_file()}}
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)
    print((directory / "command.log").read_text(encoding="utf-8", errors="replace")[-14000:], flush=True)


if __name__ == "__main__":
    main()
