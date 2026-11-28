"""evaluación coco común a faster rcnn y yolo: métricas, ap por clase, curvas pr y archivos de resultados

los dos scripts de evaluación generan los mismos archivos con el mismo formato, así que el notebook de
análisis y la comparación entre modelos los leen igual
"""

import csv
import json
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# nombres de las 12 métricas que imprime cocoeval.summarize
STAT_NAMES = [
    "map", "map50", "map75", "map_small", "map_medium", "map_large",
    "mar1", "mar10", "mar100", "mar_small", "mar_medium", "mar_large",
]
WARMUP_IMAGES = 3


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


def map_on_images(ground_truth, predictions, image_ids):
    """map@[.5:.95] y map@0.5 sobre una lista de imágenes que puede repetir imágenes (para el bootstrap)

    cada aparición de una imagen se renumera como una imagen nueva con sus cajas y sus predicciones
    """
    by_image = {}
    for p in predictions:
        by_image.setdefault(p["image_id"], []).append(p)
    images, annotations, detections = [], [], []
    for new_id, old_id in enumerate(image_ids, start=1):
        images.append({**ground_truth.imgs[old_id], "id": new_id})
        for a in ground_truth.imgToAnns[old_id]:
            annotations.append({**a, "image_id": new_id, "id": len(annotations) + 1})
        detections.extend({**p, "image_id": new_id} for p in by_image.get(old_id, []))
    subset = COCO()
    subset.dataset = {"images": images, "annotations": annotations, "categories": ground_truth.dataset["categories"]}
    with redirect_stdout(StringIO()):
        subset.createIndex()
        if not detections:
            return 0.0, 0.0
        evaluator, _ = coco_evaluate(subset, detections, list(range(1, len(image_ids) + 1)))
    return float(evaluator.stats[0]), float(evaluator.stats[1])


def paired_bootstrap(ground_truth, predictions_a, predictions_b, image_ids, resamples=300, seed=42):
    """intervalo del 95 % de map(a) - map(b) remuestreando las mismas imágenes para los dos"""
    rng = np.random.default_rng(seed)
    image_ids = list(image_ids)
    differences = []
    for _ in range(resamples):
        sample = list(rng.choice(image_ids, size=len(image_ids), replace=True))
        differences.append(map_on_images(ground_truth, predictions_a, sample)[0]
                           - map_on_images(ground_truth, predictions_b, sample)[0])
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "p_not_better": float(np.mean(np.array(differences) <= 0))}


def latency_summary(latencies_s):
    # descartamos las primeras imágenes, que incluyen la inicialización del modelo
    timed = np.array(latencies_s[WARMUP_IMAGES:]) * 1000
    return {
        "batch_size": 1,
        "warmup_images": WARMUP_IMAGES,
        "mean_ms": float(timed.mean()),
        "median_ms": float(np.median(timed)),
        "p95_ms": float(np.percentile(timed, 95)),
        "fps": float(1000 / timed.mean()),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results(out_dir, split, metrics, evaluator, ground_truth, predictions):
    """escribe <split>_metrics.json, _per_class.csv, _pr_curves.csv y _predictions.json"""
    out_dir.mkdir(parents=True, exist_ok=True)
    # las métricas coco van antes del bloque de inferencia, como en los json ya guardados
    inference = metrics["inference"]
    metrics = {**{k: v for k, v in metrics.items() if k != "inference"}, **{name: float(value) for name, value in zip(STAT_NAMES, evaluator.stats)},
               "inference": inference}
    (out_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_csv(out_dir / f"{split}_per_class.csv", per_class_metrics(evaluator, ground_truth))
    write_csv(out_dir / f"{split}_pr_curves.csv", pr_curves(evaluator, ground_truth))
    (out_dir / f"{split}_predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    return metrics
