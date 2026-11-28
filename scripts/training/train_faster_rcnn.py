"""entrenamiento de faster rcnn v2 desde la terminal"""

import argparse
from pathlib import Path

# usamos las mismas funciones de entrenamiento que se importan desde el notebook
from parasite_detection.faster_rcnn import (
    CocoDetectionDataset,
    TrainConfig,
    build_model,
    create_dataloaders,
    evaluate_loss,
    evaluate_model,
    load_trained_model,
    train_model,
)

# la terminal usa el dataset convertido si no se indica otra carpeta
from parasite_detection.paths import DATA, get_experiment_data_dir, get_experiment_run_name

# estas funciones también se pueden importar desde la carpeta de entrenamiento
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


def main():
    # argparse convierte los argumentos de la terminal a valores de python
    parser = argparse.ArgumentParser(description="Entrena Faster R-CNN v2.")
    # estos parámetros controlan las épocas y el tamaño del lote
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    # el warmup sube el learning rate antes de aplicar el descenso coseno
    parser.add_argument("--start-factor", type=float, help="Fracción inicial del learning rate en el warmup.")
    parser.add_argument("--end-factor", type=float, help="Fracción final del learning rate en el warmup.")
    parser.add_argument("--warmup-batches", type=int, help="Máximo de lotes de warmup en la primera época.")
    # patience indica cuántas épocas esperamos sin mejorar el map
    parser.add_argument("--patience", type=int, help="Usa 0 para desactivar la parada temprana.")
    # cada ejecución puede elegir sus datos y su dispositivo
    parser.add_argument("--data-dir", type=Path, default=get_experiment_data_dir(DATA))
    parser.add_argument("--device", help="Por ejemplo auto, cpu o cuda.")
    # el nombre de ejecución identifica sus pesos y sus métricas
    parser.add_argument("--run-name", default=get_experiment_run_name("baseline"))
    # un checkpoint permite recuperar el modelo y el estado del optimizador
    parser.add_argument("--resume", type=Path, help="Ruta a last_checkpoint.pth.")

    # usamos los valores de trainconfig para los argumentos que se hayan omitido
    options = vars(parser.parse_args())
    # la ruta de reanudación se pasa aparte de la configuración
    resume = options.pop("resume")
    config = TrainConfig(**{name: value for name, value in options.items() if value is not None})
    # el entrenamiento muestra el progreso y guarda los resultados por época
    train_model(config, resume=resume)


if __name__ == "__main__":
    main()
