"""Convert the official FingerMovements MATLAB release to model-ready NPZ.

The source is BCI Competition II, Data Set IV, 100 Hz (`sp1s_aa.mat`).
Conversion is lossless apart from storing EEG samples as float32. It does not
normalize, filter, augment, shuffle, or alter the official train/test split.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

EXPECTED_CHANNEL_NAMES = np.asarray(
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
EXPECTED_CHANNELS = 28
EXPECTED_TIMEPOINTS = 50
EXPECTED_CASES = {"train": 316, "test": 100}
EXPECTED_CLASS_COUNTS = {"train": {0: 159, 1: 157}, "test": {0: 49, 1: 51}}
EXPECTED_SHA256 = {
    "sp1s_aa.mat": "4ecb9f7bce25a67d71ade1bca68a103ba93f4173fc6e8426bc14aa1dade69f5c",
    "labels_data_set_iv.txt": "9ae9c41f237c9445ad749fc5539c2861c21a14195ecf308c8ccfdeb92f296c65",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a source file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_hash(path: Path) -> None:
    """Reject files that differ from the verified official downloads."""
    expected = EXPECTED_SHA256[path.name]
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"Unexpected SHA-256 for {path}: {actual}; expected {expected}"
        )


def matlab_channel_names(raw: np.ndarray) -> np.ndarray:
    """Convert scipy's MATLAB cell-array representation to Unicode names."""
    names: list[str] = []
    for cell in np.asarray(raw).reshape(-1):
        value = np.asarray(cell).squeeze()
        names.append(str(value.item() if value.ndim == 0 else value))
    return np.asarray(names)


def load_official_sources(
    mat_path: Path, test_label_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and transpose official time-channel-trial arrays."""
    for path in (mat_path, test_label_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required official source not found: {path}")
        validate_source_hash(path)

    matlab = loadmat(
        mat_path,
        variable_names=["clab", "x_train", "y_train", "x_test"],
    )
    required = {"clab", "x_train", "y_train", "x_test"}
    missing = required - set(matlab)
    if missing:
        raise KeyError(f"Official MATLAB file is missing variables: {sorted(missing)}")

    # Official layout is time x channels x trials. Models use
    # trials x channels x time.
    train_x = np.asarray(matlab["x_train"]).transpose(2, 1, 0).astype(np.float32)
    test_x = np.asarray(matlab["x_test"]).transpose(2, 1, 0).astype(np.float32)
    train_y = np.asarray(matlab["y_train"]).reshape(-1).astype(np.uint8)
    test_y = np.loadtxt(test_label_path, dtype=np.uint8).reshape(-1)
    channel_names = matlab_channel_names(matlab["clab"])
    return train_x, train_y, test_x, test_y, channel_names


def sample_keys(x: np.ndarray) -> set[bytes]:
    """Return exact float32 trial representations for duplicate checks."""
    contiguous = np.ascontiguousarray(x)
    return {sample.tobytes() for sample in contiguous}


def has_uea_sliding_channel_error(x: np.ndarray) -> bool:
    """Detect the deterministic 22-sample overlap in the retired UEA files."""
    return bool(np.all(x[:, :-1, 28:] == x[:, 1:, :22]))


def validate_dataset(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    channel_names: np.ndarray,
) -> None:
    """Fail fast if the official source violates the expected schema."""
    if not np.array_equal(channel_names, EXPECTED_CHANNEL_NAMES):
        raise ValueError(f"Unexpected channel order: {channel_names.tolist()}")

    for split, x, y in (
        ("train", train_x, train_y),
        ("test", test_x, test_y),
    ):
        expected_cases = EXPECTED_CASES[split]
        expected_shape = (expected_cases, EXPECTED_CHANNELS, EXPECTED_TIMEPOINTS)
        if x.shape != expected_shape:
            raise ValueError(f"{split}: x shape {x.shape}, expected {expected_shape}")
        if y.shape != (expected_cases,):
            raise ValueError(f"{split}: unexpected y shape {y.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"{split}: non-finite EEG values found")
        counts = Counter(y.tolist())
        if dict(counts) != EXPECTED_CLASS_COUNTS[split]:
            raise ValueError(f"{split}: unexpected class counts {dict(counts)}")
        if len(sample_keys(x)) != len(x):
            raise ValueError(f"{split}: exact duplicate trials found")
        if has_uea_sliding_channel_error(x):
            raise ValueError(
                f"{split}: detected the retired UEA sliding-channel layout error"
            )

    overlap = sample_keys(train_x) & sample_keys(test_x)
    if overlap:
        raise ValueError(f"train/test leakage: {len(overlap)} exact trials overlap")


def save_split(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    channel_names: np.ndarray,
) -> None:
    """Write one self-describing split without pickle-backed objects."""
    np.savez_compressed(
        path,
        x=x,
        y=y,
        source_index=np.arange(len(y), dtype=np.int32),
        channel_names=channel_names,
    )


def prepare(raw_dir: Path, output_dir: Path) -> None:
    mat_source = raw_dir / "sp1s_aa.mat"
    test_label_source = raw_dir / "labels_data_set_iv.txt"
    train_x, train_y, test_x, test_y, channel_names = load_official_sources(
        mat_source,
        test_label_source,
    )
    validate_dataset(train_x, train_y, test_x, test_y, channel_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_split(output_dir / "train.npz", train_x, train_y, channel_names)
    save_split(output_dir / "test.npz", test_x, test_y, channel_names)

    print("FingerMovements official MATLAB conversion complete")
    print(f"source SHA-256: {sha256(mat_source)}")
    for split, x, y in (
        ("train", train_x, train_y),
        ("test", test_x, test_y),
    ):
        counts = Counter(y.tolist())
        print(
            f"{split:>5}: x={x.shape} {x.dtype} | y={y.shape} {y.dtype} | "
            f"left={counts[0]} right={counts[1]}"
        )
    print("channel-layout integrity: PASS")
    print(f"output: {output_dir}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=repo_root / "data" / "raw" / "FingerMovements",
        help="Directory containing sp1s_aa.mat and official test labels.",
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
