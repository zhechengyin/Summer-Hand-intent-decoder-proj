"""Convert the canonical FingerMovements .ts files into model-ready NPZ files.

The converter is intentionally lossless apart from converting the source
decimal values to float32. It does not normalize, filter, augment, shuffle, or
change the dataset's official train/test split.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


CHANNEL_NAMES = np.asarray(
    [
        "F3",
        "F1",
        "Fz",
        "F2",
        "F4",
        "FC5",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "FC6",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP5",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "CP6",
        "O1",
        "O2",
    ]
)
LABEL_TO_ID = {"left": 0, "right": 1}
EXPECTED_CHANNELS = 28
EXPECTED_TIMEPOINTS = 50
EXPECTED_CASES = {"train": 316, "test": 100}


def parse_ts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse one equal-length, multivariate UEA .ts file."""
    metadata: dict[str, str] = {}
    samples: list[list[list[float]]] = []
    labels: list[int] = []
    in_data = False

    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if not in_data:
                if not line.startswith("@"):
                    raise ValueError(
                        f"{path}:{line_number}: expected metadata before @data"
                    )
                key, _, value = line[1:].partition(" ")
                metadata[key.lower()] = value.strip()
                if key.lower() == "data":
                    in_data = True
                continue

            fields = line.split(":")
            if len(fields) != EXPECTED_CHANNELS + 1:
                raise ValueError(
                    f"{path}:{line_number}: expected {EXPECTED_CHANNELS} "
                    f"channels plus one label, found {len(fields)} fields"
                )

            label_text = fields[-1].strip().lower()
            if label_text not in LABEL_TO_ID:
                raise ValueError(
                    f"{path}:{line_number}: unsupported label {label_text!r}"
                )

            channels: list[list[float]] = []
            for channel_index, field in enumerate(fields[:-1]):
                values = [float(value) for value in field.split(",")]
                if len(values) != EXPECTED_TIMEPOINTS:
                    raise ValueError(
                        f"{path}:{line_number}: channel {channel_index} has "
                        f"{len(values)} values, expected {EXPECTED_TIMEPOINTS}"
                    )
                channels.append(values)

            samples.append(channels)
            labels.append(LABEL_TO_ID[label_text])

    if not in_data:
        raise ValueError(f"{path}: missing @data marker")

    expected_metadata = {
        "timestamps": "false",
        "missing": "false",
        "univariate": "false",
        "dimensions": str(EXPECTED_CHANNELS),
        "equallength": "true",
        "serieslength": str(EXPECTED_TIMEPOINTS),
    }
    for key, expected in expected_metadata.items():
        actual = metadata.get(key, "").lower()
        if actual != expected:
            raise ValueError(
                f"{path}: @{key} is {actual!r}, expected {expected!r}"
            )

    x = np.asarray(samples, dtype=np.float32)
    y = np.asarray(labels, dtype=np.uint8)
    expected_shape = (len(samples), EXPECTED_CHANNELS, EXPECTED_TIMEPOINTS)
    if x.shape != expected_shape:
        raise ValueError(f"{path}: parsed shape {x.shape}, expected {expected_shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: non-finite EEG values found")
    return x, y


def sample_keys(x: np.ndarray) -> set[bytes]:
    """Return exact float32 byte representations for duplicate checks."""
    contiguous = np.ascontiguousarray(x)
    return {sample.tobytes() for sample in contiguous}


def validate_dataset(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> None:
    """Fail fast if the source archive violates the expected fixed schema."""
    for split, x, y in (
        ("train", train_x, train_y),
        ("test", test_x, test_y),
    ):
        expected_cases = EXPECTED_CASES[split]
        if x.shape != (
            expected_cases,
            EXPECTED_CHANNELS,
            EXPECTED_TIMEPOINTS,
        ):
            raise ValueError(f"{split}: unexpected x shape {x.shape}")
        if y.shape != (expected_cases,):
            raise ValueError(f"{split}: unexpected y shape {y.shape}")
        if set(np.unique(y).tolist()) != set(LABEL_TO_ID.values()):
            raise ValueError(f"{split}: expected both left and right labels")
        if len(sample_keys(x)) != len(x):
            raise ValueError(f"{split}: exact duplicate samples found")

    overlap = sample_keys(train_x) & sample_keys(test_x)
    if overlap:
        raise ValueError(f"train/test leakage: {len(overlap)} exact samples overlap")


def save_split(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    """Write one self-describing split without pickle-backed objects."""
    np.savez_compressed(
        path,
        x=x,
        y=y,
        source_index=np.arange(len(y), dtype=np.int32),
        channel_names=CHANNEL_NAMES,
    )


def prepare(raw_dir: Path, output_dir: Path) -> None:
    train_source = raw_dir / "FingerMovements_TRAIN.ts"
    test_source = raw_dir / "FingerMovements_TEST.ts"
    for source in (train_source, test_source):
        if not source.is_file():
            raise FileNotFoundError(f"Required source file not found: {source}")

    train_x, train_y = parse_ts(train_source)
    test_x, test_y = parse_ts(test_source)
    validate_dataset(train_x, train_y, test_x, test_y)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_split(output_dir / "train.npz", train_x, train_y)
    save_split(output_dir / "test.npz", test_x, test_y)

    print("FingerMovements conversion complete")
    for split, x, y in (
        ("train", train_x, train_y),
        ("test", test_x, test_y),
    ):
        counts = Counter(y.tolist())
        print(
            f"{split:>5}: x={x.shape} {x.dtype} | y={y.shape} {y.dtype} | "
            f"left={counts[0]} right={counts[1]}"
        )
    print(f"output: {output_dir}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=repo_root / "data" / "raw" / "FingerMovements",
        help="Directory containing the canonical TRAIN.ts and TEST.ts files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "processed" / "finger_movements",
        help="Directory in which train.npz and test.npz will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    prepare(arguments.raw_dir.resolve(), arguments.output_dir.resolve())
