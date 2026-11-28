"""evalúa un checkpoint de faster rcnn v2 en una partición y guarda métricas, curvas pr y predicciones"""

import argparse
import platform
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from parasite_detection.evaluation import coco_evaluate, latency_summary, save_results
from parasite_detection.faster_rcnn import CocoDetectionDataset, load_trained_model, resolve_device
from parasite_detection.dataset_validation import validate_coco_dataset
from parasite_detection.paths import (
    DEDUP_DATA,
    MODELS,
    RESULTS,
    ROOT,
    get_experiment_data_dir,
    get_experiment_run_name,
)


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


def main():
    parser = argparse.ArgumentParser(description="Evalúa Faster R-CNN v2 con COCOeval.")
    parser.add_argument("--run-name", default=get_experiment_run_name("baseline"))
    parser.add_argument("--checkpoint", type=Path, help="Por defecto models/trained/fasterrcnn_v2/<run>/best_model.pth.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", type=Path, default=get_experiment_data_dir(DEDUP_DATA))
    parser.add_argument("--device", default="auto")
    options = parser.parse_args()

    checkpoint = options.checkpoint or MODELS / "trained/fasterrcnn_v2" / options.run_name / "best_model.pth"
    device = resolve_device(options.device)
    out_dir = RESULTS / "metrics/fasterrcnn_v2" / options.run_name

    torch.manual_seed(42)
    validate_coco_dataset(options.data_dir, splits=(options.split,))
    model = load_trained_model(checkpoint, device=device)
    dataset = CocoDetectionDataset(options.data_dir, options.split)
    print(f"Dispositivo: {device} | {options.split}: {len(dataset)} imágenes | {checkpoint}")

    predictions, image_ids, latencies = predict(model, dataset, device)
    evaluator, summary = coco_evaluate(dataset.coco, predictions, image_ids)
    print(summary)

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
        "inference": {
            "device": str(device),
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "torch_threads": torch.get_num_threads(),
            **latency_summary(latencies),
        },
    }
    metrics = save_results(out_dir, options.split, metrics, evaluator, dataset.coco, predictions)
    print(f"mAP {metrics['map']:.4f} | mAP50 {metrics['map50']:.4f} | "
          f"{metrics['inference']['mean_ms']:.0f} ms/img | resultados en {out_dir}")


if __name__ == "__main__":
    main()
