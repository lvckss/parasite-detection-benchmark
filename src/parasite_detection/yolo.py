"""entrenamiento de yolov10 s con el protocolo experimental de faster rcnn"""

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import E2EDetectLoss

# compartimos la carga de coco y las funciones que ya usa el entrenamiento de rcnn
from parasite_detection.faster_rcnn import (
    TrainConfig as FasterRCNNConfig,
    create_dataloaders,
    evaluate_loss,
    evaluate_model,
    resolve_device,
    save_checkpoint,
    train_one_epoch,
)
from parasite_detection.paths import DEDUP_DATA, MODELS, RESULTS, ROOT


COCO_WEIGHTS = MODELS / "pretrained/yolov10s.pt"
NUM_CLASSES = 8
SCORE_THRESHOLD = 0.05
MAX_DETECTIONS = 100


@dataclass
class TrainConfig(FasterRCNNConfig):
    # usamos por defecto el mismo train depurado que en el experimento de rcnn
    data_dir: Path = DEDUP_DATA
    patience: int = 10

    # estas ganancias corresponden a la pérdida de detección de yolo
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5


class YOLODetector(torch.nn.Module):
    def __init__(self, image_size=640, weights_path=COCO_WEIGHTS, loss_gains=(7.5, 0.5, 1.5)):
        super().__init__()
        self.image_size = image_size

        # construimos únicamente la variante s con ocho clases de salida
        self.detector = DetectionModel("yolov10s.yaml", nc=NUM_CLASSES, verbose=False)
        if weights_path is not None:
            if not Path(weights_path).is_file():
                raise FileNotFoundError(f"Faltan los pesos en {weights_path}. Ejecuta just weights.")
            pretrained = YOLO(str(weights_path)).model
            source = pretrained.float().state_dict()
            destination = self.detector.state_dict()

            # transferimos los tensores compatibles y dejamos nuevas las salidas de clase
            compatible = {
                name: value for name, value in source.items()
                if name in destination and value.shape == destination[name].shape
            }
            self.detector.load_state_dict(compatible, strict=False)
            print(f"Pesos COCO transferidos: {len(compatible)}/{len(destination)} tensores")

        # fijamos las ganancias de la pérdida sin activar el entrenador automático
        box_gain, cls_gain, dfl_gain = loss_gains
        self.detector.args = get_cfg(overrides={"box": box_gain, "cls": cls_gain, "dfl": dfl_gain})
        self.detector.criterion = None

        # activamos explícitamente la rama final de yolov10 para la inferencia
        # algunas versiones de ultralytics la desactivan al construir desde el yaml
        self.detector.model[-1].end2end = True

        # conservamos el límite de cien detecciones por imagen que usa rcnn
        self.detector.model[-1].max_det = MAX_DETECTIONS

    def prepare_batch(self, images, targets=None):
        batch_images, geometry = [], []
        boxes, classes, batch_indices = [], [], []
        for index, image in enumerate(images):
            height, width = image.shape[-2:]
            ratio = self.image_size / max(height, width)
            new_height = max(1, round(height * ratio))
            new_width = max(1, round(width * ratio))
            scale_x, scale_y = new_width / width, new_height / height
            left = (self.image_size - new_width) // 2
            top = (self.image_size - new_height) // 2

            # ajustamos el lado mayor y rellenamos sin recortar contenido de la imagen
            resized = functional.interpolate(
                image.unsqueeze(0), size=(new_height, new_width),
                mode="bilinear", align_corners=False, antialias=True,
            ).squeeze(0)
            padded = functional.pad(
                resized,
                (left, self.image_size - new_width - left, top, self.image_size - new_height - top),
                value=114 / 255,
            )
            batch_images.append(padded)
            geometry.append((height, width, scale_x, scale_y, left, top))
            if targets is None:
                continue

            # convertimos las clases de coco de uno a ocho a las clases de yolo de cero a siete
            labels = targets[index]["labels"] - 1
            if ((labels < 0) | (labels >= NUM_CLASSES)).any():
                raise ValueError("Las categorías COCO deben estar entre 1 y 8.")
            image_boxes = targets[index]["boxes"].clone()
            image_boxes[:, 0::2] = image_boxes[:, 0::2] * scale_x + left
            image_boxes[:, 1::2] = image_boxes[:, 1::2] * scale_y + top

            # yolo recibe centro ancho y alto normalizados respecto al lienzo cuadrado
            centers = (image_boxes[:, :2] + image_boxes[:, 2:]) / 2
            sizes = image_boxes[:, 2:] - image_boxes[:, :2]
            boxes.append(torch.cat((centers, sizes), dim=1) / self.image_size)
            classes.append(labels.reshape(-1, 1).float())
            batch_indices.append(torch.full_like(labels, index))

        batch = {"img": torch.stack(batch_images)}
        if targets is not None:
            batch.update({
                "bboxes": torch.cat(boxes),
                "cls": torch.cat(classes),
                "batch_idx": torch.cat(batch_indices),
            })
        return batch, geometry

    def forward(self, images, targets=None):
        batch, geometry = self.prepare_batch(images, targets)
        if targets is not None:
            # mantenemos la pérdida dual original de yolov10 en ambas ramas de asignación
            if self.detector.criterion is None:
                self.detector.criterion = E2EDetectLoss(self.detector)
            losses, _ = self.detector.loss(batch)

            # ultralytics escala la pérdida por batch y aquí usamos su media por imagen
            return {"loss_yolo": losses.sum() / len(images)}

        # yolov10 devuelve cajas finales de la rama one to one sin aplicar nms
        predictions = self.detector(batch["img"])
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        # comprobamos el formato para no confundir mapas intermedios con detecciones finales
        if predictions.ndim != 3 or predictions.shape[0] != len(images) or predictions.shape[-1] != 6:
            raise RuntimeError(
                f"Salida de YOLOv10 no válida: {tuple(predictions.shape)}. "
                "Se esperaba un tensor con forma batch por detecciones por 6."
            )
        if not torch.isfinite(predictions).all() or ((predictions[..., 4] < 0) | (predictions[..., 4] > 1)).any():
            raise RuntimeError("YOLOv10 devolvió detecciones no finitas o confianzas fuera de [0, 1].")
        outputs = []
        for detections, (height, width, scale_x, scale_y, left, top) in zip(predictions, geometry):
            # usamos el mismo umbral de confianza que el detector de rcnn
            detections = detections[detections[:, 4] >= SCORE_THRESHOLD]
            image_boxes = detections[:, :4].clone()

            # deshacemos el relleno y el escalado antes de evaluar sobre el coco original
            image_boxes[:, 0::2] = ((image_boxes[:, 0::2] - left) / scale_x).clamp(0, width)
            image_boxes[:, 1::2] = ((image_boxes[:, 1::2] - top) / scale_y).clamp(0, height)
            valid = (image_boxes[:, 2:] > image_boxes[:, :2]).all(dim=1)
            outputs.append({
                "boxes": image_boxes[valid],
                "scores": detections[valid, 4],
                "labels": detections[valid, 5].long() + 1,
            })
        return outputs


def build_model(image_size=640, weights_path=COCO_WEIGHTS, loss_gains=(7.5, 0.5, 1.5)):
    # el adaptador permite usar las mismas funciones de entrenamiento y evaluación
    return YOLODetector(image_size, weights_path, loss_gains)


def load_trained_model(path=MODELS / "trained/yolov10s/baseline/best_model.pth", device="auto"):
    # reconstruimos la arquitectura y cargamos solo tensores sin necesitar pesos de coco
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["image_size"], weights_path=None, loss_gains=checkpoint["loss_gains"])
    model.load_state_dict(checkpoint["model"])
    return model.to(resolve_device(device)).eval()


def train_model(config, resume=None):
    # validamos los parámetros antes de crear carpetas o cargar pesos
    if config.epochs < 1 or config.batch_size < 1 or config.patience < 0 or config.num_workers < 0:
        raise ValueError("Revisa epochs, batch_size, patience y num_workers.")
    if config.image_size < 32 or config.image_size % 32:
        raise ValueError("image_size debe ser un múltiplo de 32.")
    if config.learning_rate <= 0 or config.weight_decay < 0 or not 0 <= config.momentum < 1:
        raise ValueError("Revisa los parámetros de SGD.")
    if config.warmup_batches < 0 or not 0 < config.start_factor <= 1 or not 0 <= config.end_factor <= 1:
        raise ValueError("Revisa warmup_batches, start_factor y end_factor.")
    if any(gain <= 0 for gain in (config.box_gain, config.cls_gain, config.dfl_gain)):
        raise ValueError("Las ganancias de la pérdida deben ser positivas.")
    if not config.run_name or Path(config.run_name).name != config.run_name or config.run_name in {".", ".."}:
        raise ValueError("Usa un nombre de experimento sin rutas.")

    device = resolve_device(config.device)
    model_dir = MODELS / "trained/yolov10s" / config.run_name
    metrics_dir = RESULTS / "metrics/yolov10s" / config.run_name
    last_path, best_path = model_dir / "last_checkpoint.pth", model_dir / "best_model.pth"
    history_path = metrics_dir / "training.csv"
    settings_path = metrics_dir / "settings.json"

    # una ejecución nueva no puede reemplazar checkpoints o métricas de otra ejecución
    if resume is None and (model_dir.exists() or metrics_dir.exists()):
        raise FileExistsError("Ya existe esta ejecución. Reanuda el entrenamiento o cambia run_name.")
    if resume is not None and Path(resume).resolve() != last_path.resolve():
        raise ValueError(f"Para reanudar esta ejecución, usa {last_path}.")

    # registramos la versión de ultralytics porque implementa la arquitectura y su pérdida
    data_dir = Path(config.data_dir).resolve()
    settings = asdict(config)
    settings["data_dir"] = str(data_dir.relative_to(ROOT) if data_dir.is_relative_to(ROOT) else data_dir)
    settings["device"] = str(device)
    settings["amp"] = config.amp and device.type == "cuda"
    settings["ultralytics_version"] = str(ultralytics.__version__)
    settings["torch_version"] = str(torch.__version__)
    settings["augmentation"] = "none"
    settings["criterion"] = "E2EDetectLoss"
    settings["evaluation"] = "pycocotools COCOeval bbox"
    settings["inference"] = "one2one"
    settings["architecture"] = "yolov10s"
    settings["preprocessing"] = "resize preserving aspect ratio and letterbox padding"
    settings["score_threshold"] = SCORE_THRESHOLD
    settings["max_detections"] = MAX_DETECTIONS
    settings["optimizer"] = "SGD"

    torch.manual_seed(config.seed)
    train_loader, val_loader = create_dataloaders(config)

    # comprobamos la identidad del train definitivo para evitar reintroducir imágenes eliminadas
    manifest_path = RESULTS / "visual-labels/dinov2_train_order.json"
    expected_ids = set(json.loads(manifest_path.read_text(encoding="utf-8"))["ordered_ids"])
    actual_ids = {Path(image["file_name"]).stem for image in train_loader.dataset.images}
    if len(train_loader.dataset) != 1157 or actual_ids != expected_ids:
        raise ValueError("El train debe contener exactamente las 1.157 imágenes definitivas depuradas.")
    for loader in (train_loader, val_loader):
        if sorted(loader.dataset.coco.cats) != list(range(1, NUM_CLASSES + 1)):
            raise ValueError("Las categorías COCO deben estar entre 1 y 8.")

    gains = (config.box_gain, config.cls_gain, config.dfl_gain)
    model = build_model(config.image_size, COCO_WEIGHTS if resume is None else None, gains).to(device)

    # igual que rcnn usamos sgd sin nesterov y sin acumulación automática de gradientes
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate,
        momentum=config.momentum, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=settings["amp"])
    history, best_map, best_epoch = [], -math.inf, 0

    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)

        # las métricas anteriores a la corrección de inferencia no sirven para elegir el mejor modelo
        if checkpoint["settings"].get("inference") != "one2one":
            raise ValueError(
                "Este checkpoint se guardó antes de corregir la inferencia one2one. "
                "Puedes cargar sus pesos con load_trained_model para reevaluarlos, "
                "pero repite el baseline con otro run_name para obtener un historial válido."
            )

        # reanudar exige conservar los ajustes que determinan los datos y la optimización
        keys = (
            "data_dir", "image_size", "seed", "batch_size", "learning_rate", "momentum",
            "weight_decay", "start_factor", "end_factor", "warmup_batches", "box_gain",
            "cls_gain", "dfl_gain", "amp", "ultralytics_version", "torch_version",
        )
        for key in keys:
            if settings[key] != checkpoint["settings"][key]:
                raise ValueError(f"Conserva el valor original de {key} para reanudar.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint["history"]
        best_map, best_epoch = checkpoint["best_map"], checkpoint["best_epoch"]

        # si ampliamos las épocas recalculamos la posición actual dentro de la curva coseno
        if scheduler.T_max != config.epochs:
            state = scheduler.state_dict()
            state["T_max"], state["last_epoch"] = config.epochs, len(history)
            state["_last_lr"] = [
                lr * (1 + math.cos(math.pi * len(history) / config.epochs)) / 2
                for lr in state["base_lrs"]
            ]
            scheduler.load_state_dict(state)
            for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
                group["lr"] = lr
        if not best_path.is_file():
            raise FileNotFoundError(f"Falta el mejor modelo de esta ejecución: {best_path}")
        del checkpoint

    first_epoch = len(history) + 1
    if first_epoch > config.epochs:
        raise ValueError("El checkpoint ya completó las épocas pedidas. Aumenta epochs.")
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Dispositivo: {device} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    for epoch in range(first_epoch, config.epochs + 1):
        torch.manual_seed(config.seed + epoch)
        description = f"Época {epoch}/{config.epochs}"

        # el warmup solo ocupa los primeros lotes de la primera época
        total_iters = min(config.warmup_batches, len(train_loader) - 1) if epoch == 1 else 0
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, description,
            config.start_factor, config.end_factor, total_iters,
        )
        if epoch == 1 and total_iters > 0:
            scheduler.base_lrs = [group["lr"] for group in optimizer.param_groups]

        # medimos la pérdida de validación en fp32 con batchnorm congelado como en rcnn
        val_loss = evaluate_loss(model, val_loader)

        # evaluamos con los mismos ids y el mismo evaluador coco que el otro detector
        metrics = evaluate_model(model, val_loader)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics, "lr": lr})
        print(
            f"{description} | train loss {train_loss:.4f} | val loss {val_loss:.4f}"
            f" | mAP {metrics['map']:.4f} | mAP50 {metrics['map50']:.4f} | lr {lr:.6f}"
        )
        scheduler.step()

        # guardamos los metadatos mínimos para reconstruir el modelo en otra sesión
        model_state = {"model": model.state_dict(), "image_size": config.image_size, "loss_gains": gains}
        if metrics["map"] > best_map:
            best_map, best_epoch = metrics["map"], epoch
            save_checkpoint(model_state, best_path)

        # el último checkpoint incluye todo lo necesario para continuar el entrenamiento
        save_checkpoint({
            **model_state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "history": history, "best_map": best_map,
            "best_epoch": best_epoch, "settings": settings,
        }, last_path)
        with history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[-1]))
            writer.writeheader()
            writer.writerows(history)

        # la parada temprana depende únicamente del map de validación
        if config.patience and epoch - best_epoch >= config.patience:
            print(f"Parada temprana tras {config.patience} épocas sin mejorar.")
            break

    # devolvemos el mejor modelo para que el notebook pueda evaluarlo más adelante
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    print(f"Mejor época: {best_epoch} | mAP {best_map:.4f} | {best_path}")
    return model.eval(), history
