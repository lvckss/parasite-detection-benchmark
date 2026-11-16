#!/usr/bin/env python3
"""Fetch the three original Hugging Face splits; uses only the standard library."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dataset.json"
RAW = ROOT / "data/raw/parasites-1s07h"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(output_dir, local_dir):
    config = json.loads(CONFIG.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, spec in config["splits"].items():
        destination = output_dir / spec["file"]
        if destination.exists() and sha256(destination) == spec["sha256"]:
            print(f"{split}: already downloaded", flush=True)
            continue
        local = local_dir / spec["file"]
        temporary = destination.with_suffix(".parquet.part")
        try:
            if local.is_file() and sha256(local) == spec["sha256"]:
                print(f"{split}: using local dataset file", flush=True)
                shutil.copyfile(local, temporary)
            else:
                url = (
                    f"https://huggingface.co/datasets/{config['repository']}/resolve/"
                    f"{config['revision']}/data/{spec['file']}"
                )
                print(f"{split}: downloading from Hugging Face", flush=True)
                request = urllib.request.Request(url, headers={"User-Agent": "parasite-detection-benchmark"})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
                    shutil.copyfileobj(response, target)
            if sha256(temporary) != spec["sha256"]:
                raise ValueError(f"Checksum mismatch for {split}; the file is incomplete or differs from the pinned dataset.")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"{split}: saved {destination.name}", flush=True)
    (output_dir / "source.json").write_text(json.dumps(config, indent=2) + "\n")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RAW)
    parser.add_argument("--local-dir", type=Path, default=ROOT / "parasites/data",
                        help="Reuse real local Parquet files; Git LFS pointers trigger a download instead.")
    args = parser.parse_args()
    try:
        print(f"Dataset ready: {download(args.output_dir.resolve(), args.local_dir.resolve())}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Download failed: {error}\nRerun this command to reuse completed files.\n")


if __name__ == "__main__":
    main()
