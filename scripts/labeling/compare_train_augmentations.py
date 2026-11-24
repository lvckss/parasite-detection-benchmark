#!/usr/bin/env python3
"""Serve a read-only browser for original and augmented training images."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/preprocessed/parasites-1s07h-augmented"
ANNOTATIONS_FILE = DATA_DIR / "annotations/instances_train.json"
LABEL_FILE = ROOT / "results/visual-labels/train_dedup_labels.json"
WEB_DIR = Path(__file__).resolve().parent / "augmentation_viewer"

LABELS = {
    "microscopy": "Microscopía",
    "collage": "Collage",
    "slide": "Diapositiva",
    "illustration": "Ilustración",
    "uncertain": "Dudosa",
}


def load_records() -> tuple[list[dict], dict[str, Path]]:
    if not ANNOTATIONS_FILE.is_file():
        raise SystemExit(
            f"No se encuentra el dataset aumentado: {ANNOTATIONS_FILE}\n"
            "Ejecuta python scripts/create_augmented_dataset.py primero."
        )

    coco = json.loads(ANNOTATIONS_FILE.read_text(encoding="utf-8"))
    originals = {}
    variants_by_source = {}
    image_paths = {}
    for image in coco["images"]:
        filename = image["file_name"]
        path = DATA_DIR / "images/train" / filename
        if not path.is_file():
            raise SystemExit(f"Falta una imagen del dataset aumentado: {path}")
        if filename in image_paths:
            raise SystemExit(f"Nombre de imagen duplicado en COCO: {filename}")
        image_paths[filename] = path

        if "source_image_id" in image:
            if "applied_transforms" not in image:
                raise SystemExit(
                    "El dataset no tiene metadatos de transformaciones. "
                    "Ejecuta python scripts/create_augmented_dataset.py "
                    "--write-transform-metadata."
                )
            variants_by_source.setdefault(image["source_image_id"], []).append(image)
        else:
            originals[image["id"]] = image

    if not originals or len(variants_by_source) != len(originals):
        raise SystemExit("El split train aumentado no tiene originales y variantes válidas.")

    records = []
    for source_id, original in originals.items():
        variants = variants_by_source.get(source_id, [])
        variants.sort(key=lambda image: image["augmentation_index"])
        if not variants:
            raise SystemExit(f"No hay variantes aumentadas para {original['file_name']}.")
        for variant in variants:
            if variant["source_image_id"] != source_id:
                raise SystemExit(f"Variante asociada a un origen incorrecto: {variant['file_name']}")
        records.append({
            "id": original["file_name"],
            "stem": Path(original["file_name"]).stem,
            "variants": [
                {
                    "id": variant["file_name"],
                    "index": variant["augmentation_index"],
                    "transforms": variant["applied_transforms"],
                }
                for variant in variants
            ],
        })

    return records, image_paths


class Handler(BaseHTTPRequestHandler):
    records: list[dict] = []
    image_paths: dict[str, Path] = {}

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, payload: object) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_labels(self) -> dict[str, str]:
        if not LABEL_FILE.exists():
            return {}
        payload = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
        labels = payload.get("labels")
        if not isinstance(labels, dict):
            raise ValueError(f"El archivo de clasificación no contiene un mapa de etiquetas: {LABEL_FILE}")
        return {
            image_id: category
            for image_id, category in labels.items()
            if isinstance(image_id, str)
            and isinstance(category, str)
            and category in LABELS
        }

    def do_GET(self) -> None:
        route = urlparse(self.path)
        if route.path == "/api/images":
            labels = self.read_labels()
            records = [
                {**record, "label": labels.get(record["stem"])}
                for record in self.records
            ]
            self.send_json({"images": records, "labels": LABELS})
            return

        if route.path.startswith("/image/"):
            image_id = unquote(route.path.removeprefix("/image/"))
            path = self.image_paths.get(image_id)
            if path is None:
                self.send_error(404)
                return
            self.send_file(path)
            return

        name = "index.html" if route.path in {"/", "/index.html"} else route.path.lstrip("/")
        candidate = (WEB_DIR / unquote(name)).resolve()
        if WEB_DIR.resolve() not in candidate.parents or not candidate.is_file():
            self.send_error(404)
            return
        self.send_file(candidate)

    def send_file(self, path: Path) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        with path.open("rb") as file:
            while chunk := file.read(128 * 1024):
                self.wfile.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    Handler.records, Handler.image_paths = load_records()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Comparador listo: {url}")
    print(f"Imágenes originales: {len(Handler.records)} | Etiquetas: {LABEL_FILE.relative_to(ROOT)}")
    print("Pulsa Ctrl+C para detenerlo.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
