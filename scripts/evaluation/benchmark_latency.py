"""mide la latencia de inferencia de varios detectores en la misma máquina y en las mismas imágenes

los modelos se alternan imagen a imagen (y cambian de orden en cada imagen), así que si la cpu se
ralentiza por temperatura o energía, afecta a todos por igual. se resume con la mediana
"""

import argparse
import csv
import json
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

# ultralytics limita torch a 8 hilos al importarse; guardamos el valor por defecto para usarlo en todos
TORCH_DEFAULT_THREADS = torch.get_num_threads()

from parasite_detection import faster_rcnn, yolo  # noqa: E402
from parasite_detection.analysis import load_coco  # noqa: E402
from parasite_detection.paths import DEDUP_DATA, MODELS, RESULTS  # noqa: E402

WARMUP_IMAGES = 3


class FasterRCNN:
    def __init__(self, run_name, device):
        self.name = f"fasterrcnn_v2/{run_name}"
        self.device = device
        self.model = faster_rcnn.load_trained_model(
            MODELS / "trained/fasterrcnn_v2" / run_name / "best_model.pth", device=device)

    @torch.inference_mode()
    def __call__(self, image):
        # igual que evaluate_faster_rcnn: la llamada al modelo incluye su redimensionado,
        # normalización y nms; el tensor ya está en memoria
        tensor = to_tensor(image).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        self.model([tensor])
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start


class YOLOv10:
    def __init__(self, run_name, device, image_size=640):
        self.name = f"yolov10s/{run_name}"
        self.device = "cpu" if device.type == "cpu" else str(device.index or 0)
        self.image_size = image_size
        self.model = yolo.load_trained_model(
            MODELS / "trained/yolov10s" / run_name / "weights/best.pt", device=str(device))

    def __call__(self, image):
        # imagen bgr en memoria: letterbox, red y salida one-to-one, igual que evaluate_yolo
        array = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        result = self.model.predict(array, imgsz=self.image_size, conf=yolo.SCORE_THRESHOLD,
                                    max_det=yolo.MAX_DETECTIONS, rect=False, device=self.device,
                                    verbose=False)[0]
        return sum(result.speed.values()) / 1000


BUILDERS = {"fasterrcnn_v2": FasterRCNN, "yolov10s": YOLOv10}


def summary(values_s):
    values = np.array(values_s) * 1000
    return {
        "images": len(values),
        "median_ms": float(np.median(values)),
        "p25_ms": float(np.percentile(values, 25)),
        "p75_ms": float(np.percentile(values, 75)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(values.mean()),
        "fps_median": float(1000 / np.median(values)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compara la latencia de varios detectores en la misma máquina.")
    parser.add_argument("--model", action="append", metavar="TIPO:RUN",
                        help="Por ejemplo fasterrcnn_v2:baseline o yolov10s:baseline. Repetible.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--images", type=int, default=50, help="Imágenes medidas por modelo, sin contar el calentamiento.")
    parser.add_argument("--data-dir", type=Path, default=DEDUP_DATA)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=TORCH_DEFAULT_THREADS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Por defecto results/metrics/latency/<split>_<device>.json.")
    options = parser.parse_args()
    specs = options.model or ["fasterrcnn_v2:baseline", "yolov10s:baseline"]

    device = faster_rcnn.resolve_device(options.device)
    models = [BUILDERS[kind](run, device) for kind, run in (spec.split(":", 1) for spec in specs)]

    # misma muestra de imágenes para todos los modelos, fijada por la semilla
    coco = load_coco(options.data_dir, options.split)
    records = sorted(coco.imgs.values(), key=lambda r: r["id"])
    random.Random(options.seed).shuffle(records)
    records = records[:WARMUP_IMAGES + options.images]
    image_dir = Path(options.data_dir) / "images" / options.split

    times = {m.name: [] for m in models}
    rows = []
    for index, record in enumerate(records):
        with Image.open(image_dir / record["file_name"]) as file:
            image = file.convert("RGB")
        # alternamos el orden para que ningún modelo vaya siempre primero
        order = models if index % 2 == 0 else models[::-1]
        for model in order:
            elapsed = model(image)
            if index == 0:
                # ultralytics vuelve a fijar sus hilos al crear el predictor en la primera llamada
                torch.set_num_threads(options.threads)
            if index >= WARMUP_IMAGES:
                times[model.name].append(elapsed)
                rows.append({"image_id": record["id"], "model": model.name, "ms": round(elapsed * 1000, 2)})
        print(f"\r{index + 1}/{len(records)} imágenes", end="", flush=True)
    print()

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "split": options.split,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "batch_size": 1,
        "warmup_images": WARMUP_IMAGES,
        "seed": options.seed,
        "protocol": "modelos alternados imagen a imagen, imagen ya en memoria, mediana",
        "models": {name: summary(values) for name, values in times.items()},
    }
    output = options.output or RESULTS / "metrics/latency" / f"{options.split}_{device.type}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, stats in result["models"].items():
        print(f"{name}: mediana {stats['median_ms']:.0f} ms/img (p25 {stats['p25_ms']:.0f}, p75 {stats['p75_ms']:.0f})")
    print(f"Resultados en {output}")


if __name__ == "__main__":
    main()
