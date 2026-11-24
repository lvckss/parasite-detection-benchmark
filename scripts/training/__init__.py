"""utilidades reutilizables para entrenar detectores"""

# reunimos las funciones públicas para poder importarlas desde esta carpeta
from .train_faster_rcnn import (
    CocoDetectionDataset,
    TrainConfig,
    build_model,
    create_dataloaders,
    evaluate_loss,
    evaluate_model,
    load_trained_model,
    train_model,
)

__all__ = [
    "CocoDetectionDataset",
    "TrainConfig",
    "build_model",
    "create_dataloaders",
    "evaluate_loss",
    "evaluate_model",
    "load_trained_model",
    "train_model",
]
