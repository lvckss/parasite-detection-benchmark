#!/usr/bin/env python3
"""Local browser tool for visual classification of deduplicated training images."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data/preprocessed/parasites-1s07h-dedup/images/train"
LABEL_FILE = ROOT / "results/visual-labels/train_dedup_labels.json"
THUMB_DIR = ROOT / ".cache/visual-labels/dedup-thumbs"
CATEGORIES = ["microscopy", "collage", "slide", "illustration", "uncertain"]
WEB_DIR = ROOT / "scripts/visual_labeler"


def image_records() -> list[dict[str, str]]:
    if not IMAGE_DIR.is_dir():
        raise SystemExit(
            f"No se encuentra el split train deduplicado: {IMAGE_DIR}\n"
            "Ejecuta antes python scripts/prepare_dataset.py y python scripts/deduplicate_train.py"
        )
    return [
        {"id": path.stem, "path": path.relative_to(ROOT).as_posix()}
        for path in sorted(IMAGE_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]


class Handler(BaseHTTPRequestHandler):
    records: list[dict[str, str]] = []
    record_by_id: dict[str, dict[str, str]] = {}
    lock = threading.Lock()

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, payload: object, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_labels(self) -> dict[str, str]:
        if LABEL_FILE.exists():
            try:
                raw = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
                labels = raw.get("labels", {})
                if isinstance(labels, dict):
                    return {
                        image_id: category
                        for image_id, category in labels.items()
                        if image_id in self.record_by_id and category in CATEGORIES
                    }
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
        return {}

    def write_labels(self, labels: dict[str, str]) -> None:
        LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": "Francesco/parasites-1s07h",
            "variant": "deduplicated",
            "split": "train",
            "categories": CATEGORIES,
            "images": len(self.records),
            "classified": len(labels),
            "labels": labels,
            "items": [
                {**record, "category": labels.get(record["id"])}
                for record in self.records
            ],
        }
        temporary = LABEL_FILE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(LABEL_FILE)

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path)
        if route.path == "/api/images":
            self.send_json({"images": self.records, "categories": CATEGORIES})
        elif route.path == "/api/labels":
            with self.lock:
                self.send_json({"labels": self.read_labels()})
        elif route.path.startswith("/image/"):
            self.serve_image(route.path.removeprefix("/image/"), thumbnail=True)
        elif route.path.startswith("/full/"):
            self.serve_image(route.path.removeprefix("/full/"), thumbnail=False)
        else:
            name = "index.html" if route.path in {"/", "/index.html"} else route.path.lstrip("/")
            candidate = (WEB_DIR / unquote(name)).resolve()
            if WEB_DIR.resolve() not in candidate.parents or not candidate.is_file():
                self.send_error(404)
                return
            data = candidate.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def serve_image(self, image_id: str, thumbnail: bool) -> None:
        record = self.record_by_id.get(image_id)
        if not record:
            self.send_error(404)
            return
        path = ROOT / record["path"]
        if thumbnail:
            THUMB_DIR.mkdir(parents=True, exist_ok=True)
            cached = THUMB_DIR / f"{image_id}.jpg"
            if not cached.exists() or cached.stat().st_mtime < path.stat().st_mtime:
                try:
                    from PIL import Image, ImageOps

                    with Image.open(path) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                        image.thumbnail((480, 360), Image.Resampling.LANCZOS)
                        image.save(cached, "JPEG", quality=82, optimize=True)
                except ImportError:
                    cached = path
            path = cached
        if not path.is_file():
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "image/jpeg")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        with path.open("rb") as image:
            while chunk := image.read(128 * 1024):
                self.wfile.write(chunk)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/labels":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            changes = body["changes"]
            if not isinstance(changes, dict) or any(
                image_id not in self.record_by_id or (category is not None and category not in CATEGORIES)
                for image_id, category in changes.items()
            ):
                raise ValueError("Etiquetas o imágenes no válidas")
        except (ValueError, KeyError, json.JSONDecodeError, TypeError):
            self.send_json({"error": "Solicitud de etiquetas no válida"}, 400)
            return
        with self.lock:
            labels = self.read_labels()
            for image_id, category in changes.items():
                if category is None:
                    labels.pop(image_id, None)
                else:
                    labels[image_id] = category
            self.write_labels(labels)
            self.send_json({"ok": True, "classified": len(labels)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Dirección local (por defecto 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="No abrir el navegador automáticamente")
    args = parser.parse_args()

    Handler.records = image_records()
    Handler.record_by_id = {record["id"]: record for record in Handler.records}
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Clasificador listo: {url}")
    print(f"Imágenes train: {len(Handler.records)} | Etiquetas: {LABEL_FILE.relative_to(ROOT)}")
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
