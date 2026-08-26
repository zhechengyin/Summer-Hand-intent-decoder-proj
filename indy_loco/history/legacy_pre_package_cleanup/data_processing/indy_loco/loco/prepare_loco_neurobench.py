"""Convert raw Loco MAT sessions into lossless 4 ms benchmark artifacts.

The conversion mirrors the input and target construction used by NeuroBench's
``PrimateReaching`` dataset with ``spike_sorting=False`` and
``bin_width=0.004``.  It does not fit a model or estimate preprocessing
statistics, and it never writes to the raw-data directory.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
from typing import Final

import h5py
import numpy as np

SCHEMA_VERSION: Final = "loco_neurobench_4ms_v1"
SAMPLE_INTERVAL_S: Final = 0.004
EXPECTED_CHANNELS: Final = 192
OFFICIAL_TRAIN_RATIO: Final = 0.5

SESSION_MD5: Final[dict[str, str]] = {
    "loco_20170210_03": "4cae63b58c4cb9c8abd44929216c703b",
    "loco_20170213_02": "e051a2ddfeb67f31395a8f934b6a04bf",
    "loco_20170214_02": "3f410a56706563b4ce5584c5b5c83cf2",
    "loco_20170215_02": "739b70762d838f3a1f358733c426bb02",
    "loco_20170216_02": "ec480664e7da8c6be0ba8ee709eecf8b",
    "loco_20170217_02": "bba2889a6ea20e74c8a9054e97a80dd4",
    "loco_20170227_04": "47dc8d717ac4e46af31a696422d83ed7",
    "loco_20170228_02": "79d99cd6b8db25ba0420a906350a44ff",
    "loco_20170301_05": "47342da09f9c950050c9213c3df38ea3",
    "loco_20170302_02": "ccbba097e02fa300ab5a87b27702f337",
}

OFFICIAL_NEUROBENCH_SESSIONS: Final[frozenset[str]] = frozenset(
    {
        "loco_20170210_03",
        "loco_20170215_02",
        "loco_20170301_05",
    }
)

DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = DATA_DIR / "raw" / "indy_loco" / "loco"
DEFAULT_OUTPUT_DIR = DATA_DIR / "processed" / "indy_loco" / "loco"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Process only the three sessions used by the official benchmark.",
    )
    parser.add_argument(
        "--session",
        action="append",
        choices=sorted(SESSION_MD5),
        help="Process one named session; repeat this option to select several.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate raw checksums and any existing processed artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing processed NPZ files after validating the raw source.",
    )
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify the published file hash
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_matlab_text(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1)
    return "".join(chr(int(value)) for value in values if int(value))


def read_channel_names(file: h5py.File, channel_count: int) -> np.ndarray:
    if "chan_names" not in file:
        return np.asarray(
            [f"channel_{index + 1:03d}" for index in range(channel_count)]
        )

    references = np.asarray(file["chan_names"]).reshape(-1)
    if references.size != channel_count:
        raise ValueError(
            f"chan_names contains {references.size} entries; expected {channel_count}"
        )
    return np.asarray([decode_matlab_text(file[reference]) for reference in references])


def event_upper_edge_indices(events: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the non-empty 4 ms bins using NeuroBench's upper-edge shift."""
    events = np.asarray(events, dtype=np.float64).reshape(-1)
    if events.size == 0:
        return np.empty(0, dtype=np.int64)

    # This is equivalent to ``nonzero(histogram(events, edges))[0] + 1`` but
    # avoids allocating a full-length histogram for every sorted unit.
    bin_indices = np.searchsorted(edges, events, side="right") - 1
    final_edge = events == edges[-1]
    bin_indices[final_edge] = edges.size - 2
    valid = (events >= edges[0]) & (events <= edges[-1])
    valid &= (bin_indices >= 0) & (bin_indices < edges.size - 1)
    return np.unique(bin_indices[valid] + 1)


def build_spike_presence(
    file: h5py.File,
    edges: np.ndarray,
) -> np.ndarray:
    spike_references = np.asarray(file["spikes"])
    if spike_references.ndim != 2:
        raise ValueError(
            f"spikes must be a 2-D MATLAB cell array, got {spike_references.shape}"
        )
    unit_count, channel_count = spike_references.shape
    if channel_count != EXPECTED_CHANNELS:
        raise ValueError(f"Loco session has {channel_count} channels; expected 192")

    presence = np.zeros((channel_count, edges.size), dtype=np.uint8)
    for unit_index in range(unit_count):
        for channel_index in range(channel_count):
            reference = spike_references[unit_index, channel_index]
            if not reference:
                continue
            cell = file[reference]
            if bool(cell.attrs.get("MATLAB_empty", 0)):
                continue
            upper_edges = event_upper_edge_indices(np.asarray(cell), edges)
            presence[channel_index, upper_edges] = 1
    return presence


def reach_bounds(target_position: np.ndarray) -> np.ndarray:
    """Reproduce NeuroBench target-change reach segmentation exactly."""
    target_channel_first = np.asarray(target_position, dtype=np.float32).T
    target_diff = np.diff(
        target_channel_first,
        axis=1,
        append=target_channel_first[:, -1].reshape(2, 1),
    )
    transition_indices = np.nonzero(np.sum(np.abs(target_diff), axis=0))[0]
    boundaries = np.insert(transition_indices, 0, 0)
    boundaries = np.append(boundaries, target_position.shape[0])
    bounds = np.column_stack((boundaries[:-1], boundaries[1:])).astype(np.int64)
    if bounds.size == 0 or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Invalid reach segmentation")
    return bounds


def official_reach_split(number_of_reaches: int) -> np.ndarray:
    """Return 0=train, 1=validation, 2=test for the official 50/25/25 split."""
    train_count = math.floor(OFFICIAL_TRAIN_RATIO * number_of_reaches)
    validation_count = math.floor((number_of_reaches - train_count) / 2)
    split = np.full(number_of_reaches, 2, dtype=np.uint8)
    split[:train_count] = 0
    split[train_count : train_count + validation_count] = 1
    return split


def convert_session(path: Path, expected_md5: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as file:
        required = {"t", "spikes", "cursor_pos", "target_pos"}
        missing = required.difference(file.keys())
        if missing:
            raise ValueError(f"{path.name} is missing datasets: {sorted(missing)}")

        timestamps = np.asarray(file["t"], dtype=np.float64).reshape(-1)
        if timestamps.size < 2 or not np.all(np.diff(timestamps) > 0):
            raise ValueError(f"Invalid timestamps in {path.name}")

        edges = np.arange(
            timestamps[0] - SAMPLE_INTERVAL_S,
            timestamps[-1],
            SAMPLE_INTERVAL_S,
            dtype=np.float64,
        )
        spike_presence = build_spike_presence(file, edges)
        cursor_position = np.asarray(file["cursor_pos"], dtype=np.float32).T
        target_position = np.asarray(file["target_pos"], dtype=np.float32).T
        channel_names = read_channel_names(file, spike_presence.shape[0])

    # ``np.arange`` can produce one additional endpoint for some floating-point
    # timestamp origins. NeuroBench never indexes past the kinematic length, so
    # discard only that unused trailing input sample while preserving every
    # benchmark-accessible value.
    if spike_presence.shape[1] < timestamps.size:
        raise ValueError(
            f"{path.name}: benchmark inputs ({spike_presence.shape[1]}) are shorter "
            f"than raw timestamps ({timestamps.size})"
        )
    spike_presence = spike_presence[:, : timestamps.size]
    sample_count = timestamps.size
    if cursor_position.shape != (sample_count, 2):
        raise ValueError(f"Unexpected cursor_pos shape: {cursor_position.shape}")
    if target_position.shape != (sample_count, 2):
        raise ValueError(f"Unexpected target_pos shape: {target_position.shape}")

    bounds = reach_bounds(target_position)
    split = official_reach_split(bounds.shape[0])
    velocity = np.gradient(cursor_position, axis=0).astype(np.float32)

    arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "session": np.asarray(path.stem),
        "source_md5": np.asarray(expected_md5),
        "sampling_interval_s": np.asarray(SAMPLE_INTERVAL_S, dtype=np.float32),
        "spike_presence": spike_presence,
        "timestamps_s": timestamps,
        "cursor_position": cursor_position,
        "velocity_per_sample": velocity,
        "target_position": target_position,
        "channel_names": channel_names,
        "reach_bounds": bounds,
        "reach_split": split,
        "official_train_ratio": np.asarray(OFFICIAL_TRAIN_RATIO, dtype=np.float32),
        "official_benchmark_session": np.asarray(
            path.stem in OFFICIAL_NEUROBENCH_SESSIONS
        ),
    }
    validate_arrays(arrays, expected_session=path.stem, expected_md5=expected_md5)
    return arrays


def validate_arrays(
    arrays: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    *,
    expected_session: str,
    expected_md5: str,
) -> None:
    required = {
        "schema_version",
        "session",
        "source_md5",
        "sampling_interval_s",
        "spike_presence",
        "timestamps_s",
        "cursor_position",
        "velocity_per_sample",
        "target_position",
        "channel_names",
        "reach_bounds",
        "reach_split",
        "official_train_ratio",
        "official_benchmark_session",
    }
    if set(arrays.keys()) != required:
        raise ValueError(f"Artifact key mismatch: {set(arrays.keys()) ^ required}")
    if str(np.asarray(arrays["schema_version"]).item()) != SCHEMA_VERSION:
        raise ValueError("Unexpected schema version")
    if str(np.asarray(arrays["session"]).item()) != expected_session:
        raise ValueError("Session name mismatch")
    if str(np.asarray(arrays["source_md5"]).item()) != expected_md5:
        raise ValueError("Source checksum mismatch")

    spikes = np.asarray(arrays["spike_presence"])
    timestamps = np.asarray(arrays["timestamps_s"])
    cursor = np.asarray(arrays["cursor_position"])
    velocity = np.asarray(arrays["velocity_per_sample"])
    target = np.asarray(arrays["target_position"])
    names = np.asarray(arrays["channel_names"])
    bounds = np.asarray(arrays["reach_bounds"])
    split = np.asarray(arrays["reach_split"])

    if spikes.dtype != np.uint8 or spikes.ndim != 2 or spikes.shape[0] != 192:
        raise ValueError(f"Invalid spike_presence: {spikes.shape}, {spikes.dtype}")
    if np.any((spikes != 0) & (spikes != 1)):
        raise ValueError("spike_presence must be binary")
    sample_count = spikes.shape[1]
    if timestamps.shape != (sample_count,) or not np.all(np.diff(timestamps) > 0):
        raise ValueError("Invalid timestamps")
    if cursor.shape != (sample_count, 2) or cursor.dtype != np.float32:
        raise ValueError("Invalid cursor_position")
    if velocity.shape != (sample_count, 2) or velocity.dtype != np.float32:
        raise ValueError("Invalid velocity_per_sample")
    if target.shape != (sample_count, 2) or target.dtype != np.float32:
        raise ValueError("Invalid target_position")
    if not np.isfinite(cursor).all() or not np.isfinite(velocity).all():
        raise ValueError("Kinematic arrays contain non-finite values")
    if names.shape != (192,) or len(set(names.tolist())) != 192:
        raise ValueError("Channel names must contain 192 unique entries")
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds[0, 0] != 0:
        raise ValueError("Invalid reach_bounds")
    if bounds[-1, 1] != sample_count or not np.array_equal(
        bounds[:-1, 1], bounds[1:, 0]
    ):
        raise ValueError("reach_bounds must cover the complete session without gaps")
    if split.shape != (bounds.shape[0],) or not set(np.unique(split)).issubset(
        {0, 1, 2}
    ):
        raise ValueError("Invalid reach_split")


def save_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as destination:
            np.savez_compressed(destination, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def selected_sessions(benchmark_only: bool, requested: list[str] | None) -> list[str]:
    if benchmark_only and requested:
        raise ValueError("--benchmark-only and --session cannot be used together")
    if requested:
        return sorted(set(requested))
    sessions = sorted(SESSION_MD5)
    if benchmark_only:
        return [
            session for session in sessions if session in OFFICIAL_NEUROBENCH_SESSIONS
        ]
    return sessions


def main() -> None:
    args = parse_args()
    sessions = selected_sessions(args.benchmark_only, args.session)
    args.raw_dir = args.raw_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    print("=== Loco 4 ms NeuroBench-compatible preparation ===")
    print(f"raw: {args.raw_dir}")
    print(f"processed: {args.output_dir}")
    print(f"sessions: {len(sessions)}")
    print("raw policy: read-only")

    missing = [
        session
        for session in sessions
        if not (args.raw_dir / f"{session}.mat").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing raw sessions: {', '.join(missing)}")

    raw_hashes: dict[str, str] = {}
    for session in sessions:
        raw_path = args.raw_dir / f"{session}.mat"
        print(f"checksum {session} ...", flush=True)
        observed = md5sum(raw_path)
        expected = SESSION_MD5[session]
        if observed != expected:
            raise ValueError(f"{session}: expected MD5 {expected}, observed {observed}")
        raw_hashes[session] = observed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        output_path = args.output_dir / f"{session}.npz"
        if args.validate_only:
            if output_path.is_file():
                with np.load(output_path, allow_pickle=False) as arrays:
                    validate_arrays(
                        arrays,
                        expected_session=session,
                        expected_md5=raw_hashes[session],
                    )
                print(f"valid processed: {output_path.name}")
            else:
                print(f"not processed: {output_path.name}")
            continue

        if output_path.exists() and not args.overwrite:
            with np.load(output_path, allow_pickle=False) as arrays:
                validate_arrays(
                    arrays,
                    expected_session=session,
                    expected_md5=raw_hashes[session],
                )
            print(f"keep valid existing: {output_path.name}")
            continue

        print(f"processing {session} ...", flush=True)
        arrays = convert_session(args.raw_dir / f"{session}.mat", raw_hashes[session])
        save_atomic(output_path, arrays)
        print(
            f"saved {output_path.name} | samples={arrays['spike_presence'].shape[1]:,} "
            f"| reaches={arrays['reach_bounds'].shape[0]} "
            f"| active={float(arrays['spike_presence'].mean()):.4%}"
        )

    print("complete")


if __name__ == "__main__":
    main()
