#!/usr/bin/env python3
"""Promote validated INT8 graph variants and bundles into Firmware and GUI."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from build_session_cubeai import MIDSIZE_DIR, SESSIONS
from build_session_cubeai_int8 import ENCODER_SOURCE_FILES, GRAPH_NAMES


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = REPOSITORY_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    return parser.parse_args()


def renamed_source(source: Path, old: str, new: str) -> str:
    text = source.read_text(encoding="utf-8")
    return text.replace(old.upper(), new.upper()).replace(old, new)


def main() -> None:
    workspace = parse_args().workspace.expanduser().resolve()
    generated = (
        workspace
        / "Custom-H747XIH6/Custom-H747XIH6/CM7/AI/Generated"
    )
    gui_models = workspace / "BCI-STM32-Plot/data/ai_device_sessions/models"
    if not generated.is_dir() or not gui_models.is_dir():
        raise FileNotFoundError(f"expected Firmware and GUI below {workspace}")

    for old in ENCODER_SOURCE_FILES:
        old_path = generated / old
        if old_path.exists():
            old_path.unlink()
    for session in SESSIONS:
        graph = GRAPH_NAMES[session]
        source_dir = MIDSIZE_DIR / session / "cubeai_int8" / "generated"
        for source_name in ENCODER_SOURCE_FILES:
            destination_name = source_name.replace("indy_encoder", graph)
            (generated / destination_name).write_text(
                renamed_source(source_dir / source_name, "indy_encoder", graph),
                encoding="utf-8",
            )
        bundle = MIDSIZE_DIR / session / "cubeai_int8" / f"{session}.aibundle"
        shutil.copyfile(bundle, gui_models / bundle.name)
        print(f"{session}: {graph}, {bundle.stat().st_size} bundle bytes")


if __name__ == "__main__":
    main()
