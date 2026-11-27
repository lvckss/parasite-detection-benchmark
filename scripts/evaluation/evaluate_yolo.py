"""evalúa un checkpoint de yolov10 s en una partición y guarda métricas, curvas pr y predicciones

genera los mismos archivos que evaluate_faster_rcnn.py para analizar y comparar ambos modelos igual
"""

import argparse
import platform
from pathlib import Path

import torch
from tqdm.auto import tqdm

# ultralytics limita torch a 8 hilos al importarse; guardamos el valor por defecto de torch, el mismo
# con el que se cronometra faster rcnn, para medir ambos modelos en igualdad de condiciones
TORCH_DEFAULT_THREADS = torch.get_num_threads()

from parasite_detection import yolo  # noqa: E402
from parasite_detection.analysis import load_coco
from parasite_detection.evaluation import coco_evaluate, latency_summary, save_results
from parasite_detection.paths import DEDUP_DATA, MODELS, RESULTS, ROOT


def measure_latency(model, image_paths, image_size, device, threads):
    # pasada aparte con lote 1: preprocesado, red y postprocesado, sin leer la imagen de disco,
    # igual que la llamada al modelo que se cronometra en faster rcnn
    predict = lambda path: model.predict(str(path), imgsz=image_size, conf=yolo.SCORE_THRESHOLD,
                                         max_det=yolo.MAX_DETECTIONS, rect=False, device=device,
                                         verbose=False)[0]
    # la primera llamada crea el predictor, y ultralytics vuelve a fijar sus hilos al crearlo
    predict(image_paths[0])
    torch.set_num_threads(threads)
    latencies = []
    for path in tqdm(image_paths, desc="Latencia", unit="img"):
        latencies.append(sum(predict(path).speed.values()) / 1000)
    return latencies


def relative(path):
    path = Path(path).resolve()
    return (path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).as_posix()


def main():
    parser = argparse.ArgumentParser(description="Evalúa YOLOv10-S con COCOeval.")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--checkpoint", type=Path, help="Por defecto models/trained/yolov10s/<run>/weights/best.pt.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--data-dir", type=Path, default=DEDUP_DATA)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, help="Por defecto results/metrics/yolov10s/<run>.")
    parser.add_argument("--threads", type=int, default=TORCH_DEFAULT_THREADS,
                        help="Hilos de CPU de torch; por defecto los de torch sin Ultralytics.")
    options = parser.parse_args()
    torch.set_num_threads(options.threads)

    checkpoint = options.checkpoint or MODELS / "trained/yolov10s" / options.run_name / "weights/best.pt"
    out_dir = options.out_dir or RESULTS / "metrics/yolov10s" / options.run_name
    device = yolo._device(options.device)

    torch.manual_seed(42)
    model = yolo.load_trained_model(checkpoint, device=options.device)
    coco = load_coco(options.data_dir, options.split)
    print(f"Dispositivo: {device} | {options.split}: {len(coco.imgs)} imágenes | {checkpoint}")

    # las métricas salen del mismo validador que eligió el mejor checkpoint durante el entrenamiento
    predictions, image_ids = yolo.predict_coco(model, options.data_dir, options.split,
                                               options.image_size, device=options.device)
    evaluator, summary = coco_evaluate(coco, predictions, image_ids)
    print(summary)

    image_dir = Path(options.data_dir) / "images" / options.split
    latencies = measure_latency(model, [image_dir / coco.imgs[i]["file_name"] for i in sorted(coco.imgs)],
                                options.image_size, device, options.threads)

    metrics = {
        "model": "yolov10s",
        "run_name": options.run_name,
        "split": options.split,
        "checkpoint": relative(checkpoint),
        "data_dir": relative(options.data_dir),
        "images": len(image_ids),
        "objects": len(coco.getAnnIds(imgIds=image_ids)),
        "detections": len(predictions),
        "image_size": options.image_size,
        "box_score_thresh": yolo.SCORE_THRESHOLD,
        "nms_thresh": None,
        "detections_per_img": yolo.MAX_DETECTIONS,
        "postprocess": "one2one, sin NMS",
        "inference": {
            "device": "cpu" if device == "cpu" else f"cuda:{device}",
            "hardware": torch.cuda.get_device_name(int(device)) if device != "cpu" else platform.processor(),
            "torch_threads": torch.get_num_threads(),
            **latency_summary(latencies),
        },
    }
    metrics = save_results(out_dir, options.split, metrics, evaluator, coco, predictions)
    print(f"mAP {metrics['map']:.4f} | mAP50 {metrics['map50']:.4f} | "
          f"{metrics['inference']['mean_ms']:.0f} ms/img | resultados en {out_dir}")


if __name__ == "__main__":
    main()
