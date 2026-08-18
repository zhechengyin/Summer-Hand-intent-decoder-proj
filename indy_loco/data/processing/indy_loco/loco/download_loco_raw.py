"""Resumably download and verify the ten immutable Loco MAT sessions."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

FILE_SPECS = {
    "loco_20170210_03": (1141201538, "4cae63b58c4cb9c8abd44929216c703b"),
    "loco_20170213_02": (1300238672, "e051a2ddfeb67f31395a8f934b6a04bf"),
    "loco_20170214_02": (1649244107, "3f410a56706563b4ce5584c5b5c83cf2"),
    "loco_20170215_02": (779509843, "739b70762d838f3a1f358733c426bb02"),
    "loco_20170216_02": (1341792566, "ec480664e7da8c6be0ba8ee709eecf8b"),
    "loco_20170217_02": (1156234641, "bba2889a6ea20e74c8a9054e97a80dd4"),
    "loco_20170227_04": (1433235546, "47dc8d717ac4e46af31a696422d83ed7"),
    "loco_20170228_02": (1599747516, "79d99cd6b8db25ba0420a906350a44ff"),
    "loco_20170301_05": (902210274, "47342da09f9c950050c9213c3df38ea3"),
    "loco_20170302_02": (1963627958, "ccbba097e02fa300ab5a87b27702f337"),
}

DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = DATA_DIR / "raw" / "indy_loco" / "loco"
URL_TEMPLATE = "https://zenodo.org/api/records/3854034/files/{filename}/content"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--session",
        action="append",
        choices=sorted(FILE_SPECS),
        help="Download one named session; repeat to select several. Default: all.",
    )
    parser.add_argument("--session-workers", type=int, default=4)
    parser.add_argument("--connections-per-session", type=int, default=4)
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - verifies Zenodo's published checksum
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_range(
    *,
    url: str,
    start: int,
    end: int,
    destination: Path,
    attempts: int = 8,
) -> None:
    expected_size = end - start + 1
    if destination.is_file() and destination.stat().st_size == expected_size:
        return

    for attempt in range(1, attempts + 1):
        try:
            request = Request(  # noqa: S310 - fixed HTTPS Zenodo endpoint
                url,
                headers={"Range": f"bytes={start}-{end}"},
            )
            with urlopen(request, timeout=120) as response:  # noqa: S310
                if response.status != 206:
                    raise OSError(f"Expected HTTP 206, received {response.status}")
                with destination.open("wb") as output:
                    shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
            if destination.stat().st_size != expected_size:
                raise OSError(
                    f"Range size mismatch: expected {expected_size}, "
                    f"received {destination.stat().st_size}"
                )
            return
        except Exception:
            destination.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(min(30, 2**attempt))


def download_session(session: str, output_dir: Path, connections: int) -> str:
    expected_size, expected_md5 = FILE_SPECS[session]
    filename = f"{session}.mat"
    final_path = output_dir / filename
    partial_path = output_dir / f"{filename}.partial"

    if final_path.is_file():
        if (
            final_path.stat().st_size != expected_size
            or md5sum(final_path) != expected_md5
        ):
            raise ValueError(f"Existing raw file failed validation: {final_path}")
        return f"valid existing: {filename}"

    prefix_size = partial_path.stat().st_size if partial_path.exists() else 0
    if prefix_size > expected_size:
        raise ValueError(f"Partial file is larger than published size: {partial_path}")

    remaining = expected_size - prefix_size
    if remaining:
        chunk_size = math.ceil(remaining / connections)
        ranges = []
        for index in range(connections):
            start = prefix_size + index * chunk_size
            if start >= expected_size:
                break
            end = min(expected_size - 1, start + chunk_size - 1)
            segment = partial_path.with_name(f"{partial_path.name}.segment-{index:02d}")
            ranges.append((start, end, segment))

        url = URL_TEMPLATE.format(filename=filename)
        with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
            futures = [
                executor.submit(
                    download_range,
                    url=url,
                    start=start,
                    end=end,
                    destination=segment,
                )
                for start, end, segment in ranges
            ]
            for future in futures:
                future.result()

        with partial_path.open("ab") as destination:
            for _, _, segment in ranges:
                with segment.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
                segment.unlink()

    if partial_path.stat().st_size != expected_size:
        raise ValueError(f"Completed size mismatch: {partial_path}")
    observed_md5 = md5sum(partial_path)
    if observed_md5 != expected_md5:
        raise ValueError(
            f"{session}: expected MD5 {expected_md5}, observed {observed_md5}"
        )
    os.replace(partial_path, final_path)
    return f"downloaded and verified: {filename}"


def main() -> None:
    args = parse_args()
    if args.session_workers < 1 or args.connections_per_session < 1:
        raise ValueError("Worker and connection counts must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions = sorted(set(args.session or FILE_SPECS))
    print("=== Loco raw downloader ===")
    print(f"destination: {output_dir}")
    print(f"sessions: {len(sessions)}")
    print(
        f"parallelism: {min(args.session_workers, len(sessions))} sessions x "
        f"{args.connections_per_session} ranges"
    )

    with ThreadPoolExecutor(
        max_workers=min(args.session_workers, len(sessions))
    ) as executor:
        future_by_session = {
            session: executor.submit(
                download_session,
                session,
                output_dir,
                args.connections_per_session,
            )
            for session in sessions
        }
        for session, future in future_by_session.items():
            print(f"{session}: {future.result()}", flush=True)
    print("complete")


if __name__ == "__main__":
    main()
