"""entrenamiento de faster rcnn v2"""

import csv
import json
import math
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from tqdm.auto import tqdm

from parasite_detection.paths import MODELS, RESULTS, ROOT


COCO_WEIGHTS = MODELS / "pretrained/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
NUM_CLASSES = 9


@dataclass
class TrainConfig:
    data_dir: Path
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 0.0025
    start_factor: float = 0.001
    end_factor: float = 1.0
    warmup_batches: int = 1000
    momentum: float = 0.9
    weight_decay: float = 0.0005
    image_size: int = 640
    patience: int = 5
    num_workers: int = 0
    seed: int = 42
    amp: bool = True
    device: str = "auto"
    run_name: str = "baseline"


class CocoDetectionDataset(Dataset):
    def __init__(self, data_dir, split):
        data_dir = Path(data_dir)
        annotations = data_dir / "annotations" / f"instances_{split}.json"
        with redirect_stdout(StringIO()):
            self.coco = COCO(str(annotations))
        self.images = self.coco.dataset["images"]
        self.image_dir = data_dir / "images" / split

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        record = self.images[index]
        with Image.open(self.image_dir / record["file_name"]) as file:
            image = to_tensor(file.convert("RGB"))

        # convertimos las cajas de coco a las dos esquinas que espera torchvision
        image_id = record["id"]
        boxes, labels = [], []
        for annotation in self.coco.imgToAnns[image_id]:
            x, y, width, height = annotation["bbox"]
            boxes.append([x, y, x + width, y + height])
            labels.append(annotation["category_id"])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
        }
        return image, target


def collate_batch(samples):
    # dejamos cada imagen y sus cajas en su propio tensor
    images, targets = zip(*samples)
    return list(images), list(targets)


def resolve_device(device):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def create_dataloaders(config, splits=("train", "val")):
    loaders = []
    for split in splits:
        dataset = CocoDetectionDataset(config.data_dir, split)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=resolve_device(config.device).type == "cuda",
            collate_fn=collate_batch,
        )
        loaders.append(loader)
    return tuple(loaders)


def build_model(image_size=640, weights_path=COCO_WEIGHTS):
    if weights_path is not None and not Path(weights_path).is_file():
        raise FileNotFoundError(
            f"Faltan los pesos en {weights_path}. "
            "Ejecuta python scripts/models/download_weights.py --only fasterrcnn."
        )
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        num_classes=91 if weights_path is not None else NUM_CLASSES,
        min_size=image_size,
        max_size=image_size,
    )
    if weights_path is not None:
        # cargamos los pesos de coco y sustituimos la cabeza por las clases del proyecto
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(features, NUM_CLASSES)
    return model


def train_one_epoch(
    model, loader, optimizer, scaler, device, description,
    start_factor=0.001, end_factor=1.0, total_iters=0,
):
    model.train()
    # el warmup sube el learning rate de forma lineal durante los primeros lotes
    warmup = None
    if total_iters > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters
        )
    total_loss = 0.0
    with tqdm(loader, desc=description, unit="lote", leave=False) as progress:
        for step, (images, targets) in enumerate(progress, start=1):
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                losses = model(images, targets)
                loss = sum(losses.values())
            batch_loss = loss.detach().item()
            if not math.isfinite(batch_loss):
                raise RuntimeError(f"Pérdida no finita en el lote {step}: {batch_loss}")

            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # si amp descarta el paso tampoco avanzamos el warmup
            if warmup is not None and scaler.get_scale() >= previous_scale:
                warmup.step()
            total_loss += batch_loss
            progress.set_postfix(
                perdida=f"{total_loss / step:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}"
            )
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_loss(model, loader):
    device = next(model.parameters()).device
    previous_modes = [(layer, layer.training) for layer in model.modules()]
    # faster rcnn devuelve las pérdidas cuando está en modo entrenamiento
    model.train()
    # mantenemos las estadísticas de batchnorm durante la validación
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.eval()

    total_loss = 0.0
    try:
        with tqdm(loader, desc="Pérdida de validación", unit="lote", leave=False) as progress:
            for step, (images, targets) in enumerate(progress, start=1):
                images = [image.to(device) for image in images]
                targets = [
                    {key: value.to(device) for key, value in target.items()}
                    for target in targets
                ]
                losses = model(images, targets)
                batch_loss = sum(losses.values()).item()
                if not math.isfinite(batch_loss):
                    raise RuntimeError(f"Pérdida de validación no finita en el lote {step}.")
                total_loss += batch_loss
                progress.set_postfix(perdida=f"{total_loss / step:.4f}")
    finally:
        for layer, was_training in previous_modes:
            layer.training = was_training
    return total_loss / len(loader)


@torch.inference_mode()
def evaluate_model(model, loader):
    model.eval()
    device = next(model.parameters()).device
    predictions, image_ids = [], []
    for images, targets in tqdm(loader, desc="Evaluación", unit="lote", leave=False):
        outputs = model([image.to(device) for image in images])
        for target, output in zip(targets, outputs):
            image_id = target["image_id"].item()
            image_ids.append(image_id)
            boxes = output["boxes"].cpu()
            # coco evalúa las cajas con ancho y alto en lugar de la segunda esquina
            boxes[:, 2:] -= boxes[:, :2]
            for box, label, score in zip(
                boxes.tolist(), output["labels"].tolist(), output["scores"].tolist()
            ):
                predictions.append(
                    {"image_id": image_id, "category_id": label, "bbox": box, "score": score}
                )

    ground_truth = loader.dataset.coco
    with redirect_stdout(StringIO()):
        if predictions:
            detections = ground_truth.loadRes(predictions)
        else:
            detections = COCO()
            detections.dataset = {**ground_truth.dataset, "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {"map": float(evaluator.stats[0]), "map50": float(evaluator.stats[1])}


def save_checkpoint(checkpoint, path):
    temporary_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_trained_model(
    path=MODELS / "trained/fasterrcnn_v2/baseline/best_model.pth", device="auto"
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["image_size"], weights_path=None)
    model.load_state_dict(checkpoint["model"])
    return model.to(resolve_device(device)).eval()


def train_model(config, resume=None):
    if config.epochs < 1 or config.batch_size < 1 or config.image_size < 1 or config.patience < 0:
        raise ValueError("Revisa epochs, batch_size, image_size y patience en la configuración.")
    if config.warmup_batches < 0 or not 0 < config.start_factor <= 1 or not 0 <= config.end_factor <= 1:
        raise ValueError("Revisa warmup_batches y los factores de LinearLR que deben estar entre 0 y 1.")
    if not config.run_name or Path(config.run_name).name != config.run_name or config.run_name == "..":
        raise ValueError("Usa un nombre de experimento sin rutas.")

    device = resolve_device(config.device)
    model_dir = MODELS / "trained" / "fasterrcnn_v2" / config.run_name
    last_path = model_dir / "last_checkpoint.pth"
    best_path = model_dir / "best_model.pth"
    metrics_dir = RESULTS / "metrics" / "fasterrcnn_v2" / config.run_name
    history_path = metrics_dir / "training.csv"
    settings_path = metrics_dir / "settings.json"
    if resume is None and (last_path.exists() or best_path.exists()):
        raise FileExistsError(f"Ya existe {model_dir}. Reanuda la ejecución o cambia run_name.")
    if resume is not None and Path(resume).resolve() != last_path.resolve():
        raise ValueError(f"Para reanudar {config.run_name}, usa {last_path}.")

    settings = asdict(config)
    data_dir = Path(config.data_dir).resolve()
    settings["data_dir"] = str(data_dir.relative_to(ROOT) if data_dir.is_relative_to(ROOT) else data_dir)
    settings["device"] = str(device)
    settings["amp"] = config.amp and device.type == "cuda"
    history = []
    best_map = -math.inf
    best_epoch = 0

    torch.manual_seed(config.seed)
    train_loader, val_loader = create_dataloaders(config)
    model = build_model(config.image_size, COCO_WEIGHTS if resume is None else None).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    # el coseno reduce el learning rate hasta cero a lo largo del entrenamiento
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=settings["amp"])

    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        for key in ("data_dir", "image_size", "seed", "batch_size", "learning_rate", "momentum", "weight_decay"):
            if settings[key] != checkpoint["settings"][key]:
                raise ValueError(f"Conserva el valor original de {key} para reanudar este experimento.")
        model.load_state_dict(checkpoint["model"])
        # aceptamos los nombres anteriores en los checkpoints ya guardados
        saved_settings = checkpoint["settings"]
        if "warmup_iters" in saved_settings:
            saved_settings["warmup_batches"] = saved_settings["warmup_iters"]
            saved_settings["start_factor"] = saved_settings["warmup_factor"]
            saved_settings["end_factor"] = 1.0
        if "total_iters" in saved_settings:
            saved_settings["warmup_batches"] = saved_settings["total_iters"]
        if "warmup_total_iters" in saved_settings:
            saved_settings["warmup_batches"] = saved_settings["warmup_total_iters"]
        for key in ("start_factor", "end_factor", "warmup_batches"):
            if key in saved_settings and settings[key] != saved_settings[key]:
                raise ValueError(f"Conserva el valor original de {key} para reanudar este experimento.")
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler.is_enabled() and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint["history"]
        best_map = checkpoint["best_map"]
        best_epoch = checkpoint["best_epoch"]
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        # recalculamos la curva si ampliamos las épocas o retomamos un checkpoint antiguo
        if "scheduler" not in checkpoint or scheduler.T_max != config.epochs:
            state = scheduler.state_dict()
            state["T_max"] = config.epochs
            state["last_epoch"] = len(history)
            state["_last_lr"] = [
                base_lr * (1 + math.cos(math.pi * len(history) / config.epochs)) / 2
                for base_lr in state["base_lrs"]
            ]
            scheduler.load_state_dict(state)
            for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
                group["lr"] = lr
            print("Curva coseno recalculada para el número de épocas indicado.")
        del checkpoint

    first_epoch = len(history) + 1
    if first_epoch > config.epochs:
        raise ValueError(f"El checkpoint ya llegó a la época {first_epoch - 1}. Aumenta epochs.")
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Dispositivo: {device} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    for epoch in range(first_epoch, config.epochs + 1):
        torch.manual_seed(config.seed + epoch)
        description = f"Época {epoch}/{config.epochs}"
        # igual que torchvision limitamos el warmup a la primera época
        total_iters = min(config.warmup_batches, len(train_loader) - 1) if epoch == 1 else 0
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, description,
            start_factor=config.start_factor, end_factor=config.end_factor, total_iters=total_iters,
        )
        if epoch == 1 and total_iters > 0:
            # el coseno parte del learning rate alcanzado al terminar el warmup
            scheduler.base_lrs = [group["lr"] for group in optimizer.param_groups]
        val_loss = evaluate_loss(model, val_loader)
        metrics = evaluate_model(model, val_loader)
        lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics, "lr": lr
        })
        print(
            f"{description} | train loss {train_loss:.4f} | val loss {val_loss:.4f}"
            f" | mAP {metrics['map']:.4f} | mAP50 {metrics['map50']:.4f} | lr {lr:.6f}"
        )
        scheduler.step()

        model_state = {"model": model.state_dict(), "image_size": config.image_size}
        if metrics["map"] > best_map:
            best_map = metrics["map"]
            best_epoch = epoch
            save_checkpoint(model_state, best_path)

        save_checkpoint(
            {
                **model_state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "best_map": best_map,
                "best_epoch": best_epoch,
                "settings": settings,
            },
            last_path,
        )
        with history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[-1]))
            writer.writeheader()
            writer.writerows(history)

        if config.patience and epoch - best_epoch >= config.patience:
            print(f"Parada temprana tras {config.patience} épocas sin mejorar.")
            break

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    print(f"Mejor época: {best_epoch} | mAP {best_map:.4f} | {best_path}")
    return model.eval(), history
