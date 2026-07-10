#!/usr/bin/env python
"""Convenience wrapper to fetch OpenNeuro ds004022.

Tries `openneuro-py` first, then falls back to `aws s3 sync --no-sign-request`.
Both are optional; see data/README.md for manual instructions.

    python tools/download_data.py --target data/ds004022
    python tools/download_data.py --target data/ds004022 --subjects sub-01 sub-02
"""
from __future__ import annotations

import argparse
import subprocess
import sys

DATASET = "ds004022"


def via_openneuro(target: str, subjects: list[str] | None) -> bool:
    try:
        import openneuro  # type: ignore
    except ImportError:
        return False
    include = [f"{s}/*" for s in subjects] if subjects else None
    print("Downloading via openneuro-py ...")
    openneuro.download(dataset=DATASET, target_dir=target, include=include)
    return True


def via_aws(target: str, subjects: list[str] | None) -> bool:
    base = f"s3://openneuro.org/{DATASET}"
    cmds = []
    if subjects:
        # top-level metadata + each requested subject folder
        cmds.append(["aws", "s3", "cp", "--no-sign-request", f"{base}/", target,
                     "--recursive", "--exclude", "*", "--include", "*.json",
                     "--include", "*.tsv", "--include", "README", "--include", "CHANGES"])
        for s in subjects:
            cmds.append(["aws", "s3", "sync", "--no-sign-request",
                         f"{base}/{s}", f"{target}/{s}"])
    else:
        cmds.append(["aws", "s3", "sync", "--no-sign-request", base, target])
    for cmd in cmds:
        print("+", " ".join(cmd))
        if subprocess.call(cmd) != 0:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="data/ds004022")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="e.g. sub-01 sub-02 (default: all)")
    args = ap.parse_args()

    if via_openneuro(args.target, args.subjects):
        print("Done (openneuro-py).")
        return 0
    print("openneuro-py not installed; trying AWS CLI ...")
    if via_aws(args.target, args.subjects):
        print("Done (aws s3).")
        return 0
    print("Could not download automatically. See data/README.md for manual options.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
