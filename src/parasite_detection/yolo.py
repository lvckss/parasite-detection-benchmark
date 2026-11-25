"""entrenamiento oficial de yolov10 s con validación coco"""

import csv
import json
import math
import tempfile
from contextlib import redirect_stdout
from copy import copy
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import torch
import ultralytics
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Format, LetterBox
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import DetectionModel

from ultralytics.utils import ops
from parasite_detection.dataset_validation import validate_loaded_image_paths, validate_yolo_dataset
from parasite_detection.paths import (
    DEDUP_DATA,
    MODELS,
    RESULTS,
    ROOT,
    TRAIN_MANIFEST,
    get_experiment_data_dir,
    get_experiment_run_name,
)


COCO_WEIGHTS = MODELS / "pretrained/yolov10s.pt"
SCORE_THRESHOLD = 0.001
MAX_DETECTIONS = 100
SUPPORTED_VERSION = "8.4.215"


@dataclass
class TrainConfig:
    # el nombre nuevo separa esta ejecución de los experimentos con el bucle anterior
    data_dir: Path = field(default_factory=lambda: get_experiment_data_dir(DEDUP_DATA))
    run_name: str = field(default_factory=lambda: get_experiment_run_name("baseline_ultralytics"))
    epochs: int = 100
    batch_size: int = 8
    image_size: int = 640

    # ultralytics crea los grupos de sgd y gestiona la acumulación de gradientes
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    nominal_batch_size: int = 64

    # el warmup oficial distingue los biases y también ajusta el momentum
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    cos_lr: bool = True
    final_lr_factor: float = 0.01

    # usamos el criterio que selecciona la arquitectura en la versión fijada de ultralytics
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5

    # la parada temprana y el mejor checkpoint dependen del map coco de validación
    patience: int = 20
    num_workers: int = 0
    seed: int = 42
    amp: bool = True
    device: str = "auto"


def _check_version():
    # estas adaptaciones se han comprobado contra la versión de uv lock
    if ultralytics.__version__ != SUPPORTED_VERSION:
        raise RuntimeError(f"Usa Ultralytics {SUPPORTED_VERSION}, la versión fijada en uv.lock.")


def _device(device):
    if "," in str(device):
        raise ValueError("Este flujo de evaluación COCO admite una sola GPU por ejecución.")
    if device == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "0"
    if str(device).startswith("cuda:"):
        return str(device).split(":", 1)[1]
    return str(device)


def _data_spec(data_dir):
    # reconstruimos las rutas locales sin depender de la máquina que convirtió el dataset
    data_dir = Path(data_dir).resolve()
    mapping = json.loads((data_dir / "class_mapping.json").read_text())
    if sorted((r["yolo_id"], r["coco_id"]) for r in mapping) != [(i, i + 1) for i in range(8)]:
        raise ValueError("Se esperan clases YOLO 0–7 y COCO 1–8 en class_mapping.json.")
    return {
        "path": str(data_dir), "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {r["yolo_id"]: r["name"] for r in mapping},
    }


def _check_dataset(data_dir):
    data_dir = Path(data_dir).resolve()
    spec = _data_spec(data_dir)
    expected = set(json.loads(TRAIN_MANIFEST.read_text())["ordered_ids"])
    validate_yolo_dataset(data_dir, expected, splits=("train", "val"))
    return spec


class CocoValidator(DetectionValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        # usamos las anotaciones originales y los ids estables del split evaluado
        annotation_file = Path(self.data["path"]) / "annotations" / f"instances_{self.args.split}.json"
        with redirect_stdout(StringIO()):
            self.coco = COCO(str(annotation_file))
        self.image_by_name = {Path(r["file_name"]).name: r["id"] for r in self.coco.dataset["images"]}
        self.coco_predictions = []
        self.coco_image_ids = []
        self.coco_results = {}

    def update_metrics(self, preds, batch):
        # conservamos las métricas nativas para poder comparar ambos evaluadores
        super().update_metrics(preds, batch)
        for index, pred in enumerate(preds):
            prepared = self._prepare_batch(index, batch)
            image_id = self.image_by_name[Path(prepared["im_file"]).name]
            self.coco_image_ids.append(image_id)

            # el escalado oficial deshace el letterbox antes de pasar las cajas a coco
            scaled = self.scale_preds(self._prepare_pred(pred), prepared)
            boxes = ops.xyxy2ltwh(scaled["bboxes"]).tolist()
            for box, score, label in zip(boxes, scaled["conf"].tolist(), scaled["cls"].tolist()):
                if box[2] > 0 and box[3] > 0:
                    self.coco_predictions.append({
                        "image_id": image_id, "category_id": int(label) + 1,
                        "bbox": box, "score": score,
                    })

    def get_stats(self):
        stats = super().get_stats()
        # evitamos elegir modelos con una validación incompleta o imágenes repetidas
        if len(self.coco_image_ids) != len(set(self.coco_image_ids)) or set(self.coco_image_ids) != set(self.coco.imgs):
            raise ValueError("La evaluación debe recorrer una vez todas las imágenes del split COCO.")
        with redirect_stdout(StringIO()):
            if self.coco_predictions:
                detections = self.coco.loadRes(self.coco_predictions)
            else:
                detections = COCO()
                detections.dataset = {**self.coco.dataset, "annotations": []}
                detections.createIndex()
            evaluator = COCOeval(self.coco, detections, "bbox")
            evaluator.params.imgIds = self.coco_image_ids
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        self.coco_results = {"map": float(evaluator.stats[0]), "map50": float(evaluator.stats[1])}

        # fitness controla best pt y la parada temprana del entrenador oficial
        stats.update({"coco/map": self.coco_results["map"], "coco/map50": self.coco_results["map50"],
                      "fitness": self.coco_results["map"]})
        print(f"COCO | mAP {self.coco_results['map']:.4f} | mAP50 {self.coco_results['map50']:.4f}")
        return stats


class ParasiteTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        # fijamos el lienzo cuadrado también en validación para igualar la evaluación independiente
        # el modo rectangular nativo añade margen y puede cambiar el tamaño real de entrada
        dataset.rect = False
        if mode == "train":
            # augment false en model train no desactiva todos los aumentos del dataset
            # sustituimos sus transformaciones por letterbox y formateo deterministas
            dataset.augment = False
            dataset.transforms = dataset.build_transforms(hyp=copy(self.args))
            if not all(isinstance(t, (LetterBox, Format)) for t in dataset.transforms.transforms):
                raise RuntimeError("El baseline solo admite letterbox y formateo de targets.")
            validate_loaded_image_paths(self.data["path"], "train", dataset.im_files)
        return dataset

    def get_validator(self):
        # el resto del entrenamiento conserva el bucle oficial sin sustituirlo
        return CocoValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)


def _metrics_dir(trainer):
    return RESULTS / "metrics/yolov10s" / trainer.save_dir.name


def _record_settings(trainer):
    # guardamos los argumentos efectivos después de resolver dispositivo y reanudación
    directory = _metrics_dir(trainer)
    directory.mkdir(parents=True, exist_ok=True)
    args = trainer.args
    settings = {
        "engine": "ultralytics", "ultralytics_version": str(ultralytics.__version__),
        "torch_version": str(torch.__version__), "architecture": "yolov10s",
        "data_dir": str(trainer.data["path"]), "run_name": trainer.save_dir.name,
        "epochs": trainer.epochs, "batch_size": trainer.batch_size, "image_size": args.imgsz,
        "learning_rate": args.lr0, "momentum": args.momentum, "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs, "nominal_batch_size": args.nbs,
        "seed": args.seed, "device": str(trainer.device), "amp": bool(trainer.amp),
        "augmentation": {
            "online": "none",
            "offline_config": (
                json.loads((Path(trainer.data["path"]) / "augmentation_config.json").read_text(encoding="utf-8"))
                if (Path(trainer.data["path"]) / "augmentation_config.json").is_file()
                else None
            ),
        },
        "criterion": type(trainer.model.init_criterion()).__name__,
        "inference": "one2one", "evaluation": "pycocotools COCOeval bbox",
        "score_threshold": args.conf, "max_detections": args.max_det,
        "selection": "COCO mAP on validation EMA", "native_args": vars(args),
        "validation_shape": "square imgsz with rect disabled",
        "loss_reporting": "sum of native logged components, separate from the backward scalar",
    }
    (directory / "settings.json").write_text(json.dumps(settings, indent=2, default=str) + "\n", encoding="utf-8")


def _record_epoch(trainer):
    # la evaluación final del mejor modelo no representa una época adicional
    if not trainer.validator.training or not trainer.csv.is_file():
        return
    with trainer.csv.open(newline="", encoding="utf-8") as file:
        native_rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(file)]
    # una interrupción tras escribir el csv puede dejar una época que se repite al reanudar
    by_epoch = {int(row["epoch"]): row for row in native_rows if int(row["epoch"]) <= trainer.epoch + 1}
    rows = []
    for epoch in sorted(by_epoch):
        native = by_epoch[epoch]
        # registramos las componentes publicadas sin cambiar el escalar del backward
        parts = {k: float(v) for k, v in native.items() if k.startswith(("train/", "val/")) and k.endswith("loss")}
        row = {
            "epoch": int(native["epoch"]),
            "train_loss": sum(v for k, v in parts.items() if k.startswith("train/")),
            "val_loss": sum(v for k, v in parts.items() if k.startswith("val/")),
            "map": float(native["coco/map"]), "map50": float(native["coco/map50"]),
            "lr": float(native["lr/pg0"]),
            "ultralytics_map": float(native["metrics/mAP50-95(B)"]),
            "ultralytics_map50": float(native["metrics/mAP50(B)"]),
        }
        row.update(parts)
        row.update({k: float(v) for k, v in native.items() if k.startswith("lr/pg")})
        rows.append(row)

    # reconstruir el csv nativo cubre una interrupción entre callbacks
    path = _metrics_dir(trainer) / "training.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_model(weights_path=COCO_WEIGHTS):
    # la transferencia y la adaptación de clases quedan a cargo del entrenador oficial
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"Faltan los pesos en {weights_path}. Ejecuta just weights.")
    return YOLO(str(weights_path))


def load_trained_model(path=None, device="auto"):
    path = (
        Path(path) if path is not None
        else MODELS / "trained/yolov10s" / get_experiment_run_name("baseline_ultralytics") / "weights/best.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".pth":
        # conservamos la lectura de los checkpoints antiguos para comparar predicciones
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        raw = DetectionModel("yolov10s.yaml", nc=8, verbose=False)
        raw.load_state_dict({k.removeprefix("detector."): v for k, v in checkpoint["model"].items()})
        raw.names = {int(k) - 1: v for k, v in json.loads((ROOT / "configs/dataset.json").read_text())["classes"].items()}
        raw.args = vars(get_cfg())
        model = YOLO("yolov10s.yaml")
        model.model = raw
        model.overrides = {"task": "detect", "imgsz": checkpoint["image_size"]}
    else:
        model = build_model(path)
    resolved = _device(device)
    model.to(resolved if resolved in {"cpu", "mps"} else f"cuda:{resolved}")
    model.model.float().eval()
    model.model.end2end = True
    return model


def evaluate_model(model, data_dir=None, split="val", image_size=None, batch_size=8, device="auto"):
    _check_version()
    if split not in {"val", "test"}:
        raise ValueError("Evalúa val o test explícitamente.")
    data_dir = get_experiment_data_dir(DEDUP_DATA) if data_dir is None else Path(data_dir)
    spec = _data_spec(data_dir)
    validate_yolo_dataset(data_dir, splits=(split,))
    if image_size is None:
        image_size = model.overrides.get("imgsz", 640)
    # la evaluación independiente comparte el postprocesado y el evaluador del entrenamiento
    with tempfile.TemporaryDirectory(prefix="parasites-yolo-eval-") as temporary:
        directory = Path(temporary)
        data_file = directory / "dataset.yaml"
        data_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
        validator = CocoValidator(save_dir=directory, args={
            "data": str(data_file), "split": split, "imgsz": image_size, "batch": batch_size,
            "device": _device(device), "workers": 0, "conf": SCORE_THRESHOLD,
            "max_det": MAX_DETECTIONS, "nms": False, "rect": False, "plots": False, "save_json": False,
        })
        validator(model=model.model)
        return validator.coco_results


def train_model(config, resume=None, dashboard=None):
    if dashboard is None:
        return _train_model(config, resume)
    from parasite_detection.dashboard.adapters import TrainingSession

    with TrainingSession(dashboard, config) as monitor:
        model, history = _train_model(config, resume, monitor)
        monitor.finish(history, model.trainer.best)
        return model, history


def _train_model(config, resume=None, monitor=None):
    _check_version()
    if not config.run_name or Path(config.run_name).name != config.run_name or config.run_name in {".", ".."}:
        raise ValueError("Usa un nombre de experimento sin rutas.")
    if config.epochs < 1 or config.batch_size < 1 or config.patience < 0 or config.num_workers < 0:
        raise ValueError("Revisa epochs, batch_size, patience y num_workers.")
    if config.image_size < 32 or config.image_size % 32:
        raise ValueError("image_size debe ser un múltiplo de 32.")
    if config.nominal_batch_size < config.batch_size or config.warmup_epochs < 0:
        raise ValueError("Revisa nominal_batch_size y warmup_epochs.")
    if not all(math.isfinite(v) for v in (config.learning_rate, config.momentum, config.weight_decay, config.final_lr_factor)):
        raise ValueError("Los hiperparámetros deben ser finitos.")
    if config.learning_rate <= 0 or config.weight_decay < 0 or not 0 <= config.momentum < 1 or not 0 <= config.final_lr_factor <= 1:
        raise ValueError("Revisa los parámetros del optimizador y final_lr_factor.")
    spec = _check_dataset(config.data_dir)
    directory = MODELS / "trained/yolov10s" / config.run_name
    metrics_dir = RESULTS / "metrics/yolov10s" / config.run_name
    if resume is None and (directory.exists() or metrics_dir.exists()):
        raise FileExistsError("Ya existe esta ejecución. Usa otro run_name o reanuda un last.pt interrumpido.")
    if resume is not None:
        if Path(resume).resolve() != (directory / "weights/last.pt").resolve():
            raise ValueError(f"Para reanudar esta ejecución, usa {directory / 'weights/last.pt'}.")
        saved = json.loads((metrics_dir / "settings.json").read_text())
        if saved.get("engine") != "ultralytics" or saved["ultralytics_version"] != SUPPORTED_VERSION:
            raise ValueError("Solo se reanudan ejecuciones del nuevo entrenador y de la misma versión.")
        if Path(saved["data_dir"]).resolve() != Path(config.data_dir).resolve() or saved["epochs"] != config.epochs:
            raise ValueError("Conserva el dataset y el horizonte de épocas original para reanudar.")
        # evitamos que los valores por defecto de la terminal alteren un experimento al reanudarlo
        native_keys = {
            "batch_size": "batch", "image_size": "imgsz", "learning_rate": "lr0", "momentum": "momentum",
            "weight_decay": "weight_decay", "seed": "seed", "nominal_batch_size": "nbs",
            "warmup_epochs": "warmup_epochs", "warmup_momentum": "warmup_momentum",
            "warmup_bias_lr": "warmup_bias_lr", "cos_lr": "cos_lr", "final_lr_factor": "lrf",
            "box_gain": "box", "cls_gain": "cls", "dfl_gain": "dfl", "amp": "amp",
        }
        for name, key in native_keys.items():
            if getattr(config, name) != saved["native_args"][key]:
                raise ValueError(f"Conserva {name}={saved['native_args'][key]} para reanudar esta ejecución.")

    model = build_model(resume if resume is not None else COCO_WEIGHTS)
    if resume is not None and (model.ckpt.get("epoch", -1) < 0 or model.ckpt.get("optimizer") is None):
        raise ValueError("El checkpoint ya terminó y Ultralytics retiró el optimizador. Inicia otra ejecución.")
    directory.mkdir(parents=True, exist_ok=True)
    data_file = directory / "dataset.yaml"
    data_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
    model.add_callback("on_train_start", _record_settings)
    model.add_callback("on_fit_epoch_end", _record_epoch)
    if monitor is not None:
        from parasite_detection.dashboard.adapters import UltralyticsEvents

        UltralyticsEvents(monitor, metrics_dir, resume=resume is not None, evaluation={
            "label": "COCOeval · one-to-one · val", "evaluator": "pycocotools COCOeval",
            "split": "val", "iou": "0.50:0.95", "max_dets": [1, 10, 100],
            "inference": "one2one", "nms": False, "score_threshold": SCORE_THRESHOLD,
            "max_detections": MAX_DETECTIONS, "weights": "EMA",
            "ultralytics_version": SUPPORTED_VERSION,
            "precision_recall": "métricas nativas de Ultralytics, no COCOeval",
            "loss_reporting": "suma de componentes nativas publicadas, no escalar del backward",
        }).install(model)

    # los ceros documentan el baseline y el dataloader impide aplicar aumentos aleatorios
    model.train(
        trainer=ParasiteTrainer, data=str(data_file), project=str(directory.parent), name=config.run_name,
        exist_ok=True, resume=resume is not None, epochs=config.epochs, batch=config.batch_size,
        imgsz=config.image_size, device=_device(config.device), workers=config.num_workers,
        seed=config.seed, deterministic=True, amp=config.amp, optimizer="SGD",
        lr0=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay,
        nbs=config.nominal_batch_size, warmup_epochs=config.warmup_epochs,
        warmup_momentum=config.warmup_momentum, warmup_bias_lr=config.warmup_bias_lr,
        cos_lr=config.cos_lr, lrf=config.final_lr_factor, patience=config.patience,
        box=config.box_gain, cls=config.cls_gain, dfl=config.dfl_gain,
        conf=SCORE_THRESHOLD, max_det=MAX_DETECTIONS, nms=False, val=True, split="val",
        plots=False, save=True, save_json=False, fraction=1.0, close_mosaic=0,
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, bgr=0.0, multi_scale=0.0, auto_augment=None, erasing=0.0,
    )
    with (metrics_dir / "training.csv").open(newline="", encoding="utf-8") as file:
        history = [{k: int(v) if k == "epoch" else float(v) for k, v in row.items()} for row in csv.DictReader(file)]
    # la serialización nativa usa fp16 y la evaluación final refleja los pesos realmente guardados
    final_validation = {
        "checkpoint": str(model.trainer.best), "split": "val", "inference": "one2one",
        "score_threshold": SCORE_THRESHOLD, "max_detections": MAX_DETECTIONS,
        **model.trainer.validator.coco_results,
    }
    (metrics_dir / "best_validation.json").write_text(json.dumps(final_validation, indent=2) + "\n", encoding="utf-8")
    return model, history
