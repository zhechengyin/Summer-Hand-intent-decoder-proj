#!/usr/bin/env python3
"""Run the identical Phase-12 protocol for all six benchmark sessions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run.py"
SESSIONS = (
    "indy_20160622_01",
    "indy_20160630_01",
    "indy_20170131_02",
    "loco_20170210_03",
    "loco_20170215_02",
    "loco_20170301_05",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--remaining-five",
        action="store_true",
        help="Skip the first session, useful after the original single-session run.",
    )
    args = parser.parse_args()
    sessions = SESSIONS[1:] if args.remaining_five else SESSIONS
    for index, session in enumerate(sessions, start=1):
        print(f"\n[{index}/{len(sessions)}] {session}", flush=True)
        command = [
            sys.executable,
            str(RUNNER),
            "--session",
            session,
            "--device",
            args.device,
            "--threads",
            str(args.threads),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
