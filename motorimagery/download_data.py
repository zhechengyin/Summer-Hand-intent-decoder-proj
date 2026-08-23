"""
Download official BCI Competition III Dataset I files and true test labels.
Run:
    python download_data.py --out data

The URLs are the official BCI Competition III resources.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request


URLS = {
    "Competition_train.mat.gz":
        "https://www.bbci.de/competition/download/competition_iii/tuebingen/Competition_train.mat.gz",
    "Competition_test.mat.gz":
        "https://www.bbci.de/competition/download/competition_iii/tuebingen/Competition_test.mat.gz",
    "true_labels.txt":
        "https://www.bbci.de/competition/iii/results/tuebingen/true_labels.txt",
}


def download(url: str, dst: Path):
    if dst.exists():
        print(f"Exists, skipping: {dst}")
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"Saved: {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        download(url, args.out / name)


if __name__ == "__main__":
    main()
