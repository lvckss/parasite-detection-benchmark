#!/usr/bin/env python3
"""Descarga los pesos COCO indicados en el guion de la práctica."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
WEIGHTS = {
    "fasterrcnn": {
        "filename": "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
        "url": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
        "sha256_prefix": "dd69338a",
    },
    "yolo": {
        "filename": "yolov10s.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10s.pt",
        "sha256_prefix": None,
    },
}
CHUNK_SIZE = 1024 * 1024


def download(name: str, force: bool = False) -> bool:
    weight = WEIGHTS[name]
    destination = MODEL_DIR / weight["filename"]
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.is_file() and not force:
        print(f"Ya existe: {destination.relative_to(ROOT)} (usa --force para reemplazarlo)")
        return True

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        weight["url"],
        headers={"User-Agent": "parasite-detection-benchmark/1.0"},
    )
    digest = hashlib.sha256()
    received = 0
    started = time.monotonic()
    last_update = 0.0
    print(f"Descargando {name} desde {weight['url']}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0"))
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                now = time.monotonic()
                if now - last_update >= 1.0:
                    if total:
                        progress = received / total * 100
                        print(f"  {received / 1024**2:.1f}/{total / 1024**2:.1f} MiB ({progress:.1f}%)", end="\r", flush=True)
                    else:
                        print(f"  {received / 1024**2:.1f} MiB", end="\r", flush=True)
                    last_update = now
    except (OSError, urllib.error.URLError, TimeoutError) as error:
        raise SystemExit(f"Error al descargar {name}: {error}\nEl archivo parcial queda en {partial}") from error

    if received == 0:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"La descarga de {name} está vacía.")
    if total and received != total:
        raise SystemExit(f"Descarga incompleta de {name}: se recibieron {received} de {total} bytes.")

    actual_sha256 = digest.hexdigest()
    expected_prefix = weight["sha256_prefix"]
    if expected_prefix and not actual_sha256.startswith(expected_prefix):
        partial.unlink(missing_ok=True)
        raise SystemExit(
            f"La suma SHA-256 de {name} no coincide: esperaba el prefijo {expected_prefix}, "
            f"se obtuvo {actual_sha256[:8]}. Se eliminó el archivo descargado."
        )

    partial.replace(destination)
    elapsed = time.monotonic() - started
    print(f"\nGuardado: {destination.relative_to(ROOT)} ({received / 1024**2:.1f} MiB, {elapsed:.1f} s)")
    if expected_prefix is None:
        print(f"  SHA-256: {actual_sha256} (el guion no publica una suma de referencia)")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=tuple(WEIGHTS),
        help="Descarga solo fasterrcnn o yolo (por defecto, ambos).",
    )
    parser.add_argument("--force", action="store_true", help="Reemplaza los archivos que ya existan.")
    args = parser.parse_args()
    selected = [args.only] if args.only else list(WEIGHTS)
    already_downloaded = [download(name, force=args.force) for name in selected]
    if all(already_downloaded):
        print("\nLos pesos ya están descargados en models/.")
    else:
        print("\nListo, los pesos están en models/.")


if __name__ == "__main__":
    main()
