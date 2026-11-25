"""evalúa un checkpoint de faster rcnn v2 en una partición y guarda métricas, curvas pr y predicciones"""

import argparse
import csv
import json
import platform
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from tqdm.auto import tqdm

from parasite_detection.faster_rcnn import CocoDetectionDataset, load_trained_model, resolve_device
from parasite_detection.paths import DEDUP_DATA, MODELS, RESULTS, ROOT

# nombres de las 12 métricas que imprime cocoeval.summarize
STAT_NAMES = [
    "map", "map50", "map75", "map_small", "map_medium", "map_large",
    "mar1", "mar10", "mar100", "mar_small", "mar_medium", "mar_large",
]
WARMUP_IMAGES = 3


@torch.inference_mode()
def predict(model, dataset, device):
    # inferencia imagen a imagen para medir la latencia de una sola imagen
    predictions, image_ids, latencies = [], [], []
    for index in tqdm(range(len(dataset)), desc="Inferencia", unit="img"):
        image, target = dataset[index]
        image = image.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = model([image])[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

        image_id = target["image_id"].item()
        image_ids.append(image_id)
        boxes = output["boxes"].cpu()
        # coco evalúa las cajas como [x, y, ancho, alto]
        boxes[:, 2:] -= boxes[:, :2]
        for box, label, score in zip(boxes.tolist(), output["labels"].tolist(), output["scores"].tolist()):
            predictions.append({
                "image_id": image_id,
                "category_id": label,
                "bbox": [round(value, 2) for value in box],
                "score": round(score, 5),
            })
    return predictions, image_ids, latencies


def coco_evaluate(ground_truth, predictions, image_ids):
    with redirect_stdout(StringIO()) as log:
        detections = ground_truth.loadRes(predictions)
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator, log.getvalue()


def per_class_metrics(evaluator, ground_truth):
    # precision tiene forma [iou, recall, clase, área, maxdets]; usamos área total y 100 detecciones
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    rows = []
    for k, category_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, k]
        ap = values[values > -1].mean() if (values > -1).any() else float("nan")
        ap50 = precision[0, :, k]
        ap50 = ap50[ap50 > -1].mean() if (ap50 > -1).any() else float("nan")
        rows.append({
            "category_id": category_id,
            "name": ground_truth.cats[category_id]["name"],
            "objects": len(ground_truth.getAnnIds(catIds=[category_id], imgIds=evaluator.params.imgIds)),
            "ap": float(ap),
            "ap50": float(ap50),
        })
    return rows


def pr_curves(evaluator, ground_truth):
    # curvas pr de cada clase a iou 0.5 y 0.75 con los 101 puntos de recall de coco
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    rows = []
    for iou_index, iou in ((0, 0.5), (5, 0.75)):
        for k, category_id in enumerate(evaluator.params.catIds):
            name = ground_truth.cats[category_id]["name"]
            for recall, value in zip(evaluator.params.recThrs, precision[iou_index, :, k]):
                rows.append({"iou": iou, "category_id": category_id, "name": name,
                             "recall": round(float(recall), 2), "precision": float(value)})
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evalúa Faster R-CNN v2 con COCOeval.")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--checkpoint", type=Path, help="Por defecto models/trained/fasterrcnn_v2/<run>/best_model.pth.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", type=Path, default=DEDUP_DATA)
    parser.add_argument("--device", default="auto")
    options = parser.parse_args()

    checkpoint = options.checkpoint or MODELS / "trained/fasterrcnn_v2" / options.run_name / "best_model.pth"
    device = resolve_device(options.device)
    out_dir = RESULTS / "metrics/fasterrcnn_v2" / options.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    model = load_trained_model(checkpoint, device=device)
    dataset = CocoDetectionDataset(options.data_dir, options.split)
    print(f"Dispositivo: {device} | {options.split}: {len(dataset)} imágenes | {checkpoint}")

    predictions, image_ids, latencies = predict(model, dataset, device)
    evaluator, summary = coco_evaluate(dataset.coco, predictions, image_ids)
    print(summary)

    timed = np.array(latencies[WARMUP_IMAGES:]) * 1000
    data_dir = options.data_dir.resolve()
    metrics = {
        "model": "fasterrcnn_resnet50_fpn_v2",
        "run_name": options.run_name,
        "split": options.split,
        "checkpoint": checkpoint.resolve().relative_to(ROOT).as_posix(),
        "data_dir": (data_dir.relative_to(ROOT) if data_dir.is_relative_to(ROOT) else data_dir).as_posix(),
        "images": len(image_ids),
        "objects": len(dataset.coco.getAnnIds(imgIds=image_ids)),
        "detections": len(predictions),
        "box_score_thresh": model.roi_heads.score_thresh,
        "nms_thresh": model.roi_heads.nms_thresh,
        "detections_per_img": model.roi_heads.detections_per_img,
        **{name: float(value) for name, value in zip(STAT_NAMES, evaluator.stats)},
        "inference": {
            "device": str(device),
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "batch_size": 1,
            "warmup_images": WARMUP_IMAGES,
            "mean_ms": float(timed.mean()),
            "median_ms": float(np.median(timed)),
            "p95_ms": float(np.percentile(timed, 95)),
            "fps": float(1000 / timed.mean()),
        },
    }
    split = options.split
    (out_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_csv(out_dir / f"{split}_per_class.csv", per_class_metrics(evaluator, dataset.coco))
    write_csv(out_dir / f"{split}_pr_curves.csv", pr_curves(evaluator, dataset.coco))
    (out_dir / f"{split}_predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    print(f"mAP {metrics['map']:.4f} | mAP50 {metrics['map50']:.4f} | "
          f"{metrics['inference']['mean_ms']:.0f} ms/img | resultados en {out_dir}")


if __name__ == "__main__":
    main()
