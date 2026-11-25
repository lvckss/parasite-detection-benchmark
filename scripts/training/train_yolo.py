"""entrenamiento de yolov10 s desde la terminal"""

import argparse
from pathlib import Path

# exponemos las mismas funciones para usarlas desde terminal o desde un notebook
from parasite_detection.yolo import (
    TrainConfig,
    build_model,
    evaluate_model,
    load_trained_model,
    train_model,
)


__all__ = [
    "TrainConfig", "build_model", "evaluate_model", "load_trained_model", "train_model",
]


def main():
    parser = argparse.ArgumentParser(description="Entrena YOLOv10-S con Ultralytics y evaluación COCO.")

    # si omitimos un argumento se conserva el valor definido en trainconfig
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)

    # usamos los parámetros nativos de warmup y acumulación de ultralytics
    parser.add_argument("--nominal-batch-size", type=int)
    parser.add_argument("--warmup-epochs", type=float)
    parser.add_argument("--warmup-momentum", type=float)
    parser.add_argument("--warmup-bias-lr", type=float)
    parser.add_argument("--final-lr-factor", type=float)
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--patience", type=int, help="Usa 0 para desactivar la parada temprana.")

    # yolo requiere un tamaño de entrada múltiplo de treinta y dos
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="auto, cpu o cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)

    # las ganancias ponderan las tres componentes de la pérdida de yolo
    parser.add_argument("--box-gain", type=float)
    parser.add_argument("--cls-gain", type=float)
    parser.add_argument("--dfl-gain", type=float)
    parser.add_argument("--run-name")

    # el checkpoint se pasa aparte para recuperar también el estado del optimizador
    parser.add_argument("--resume", type=Path, help="Ruta a weights/last.pt de una ejecución interrumpida.")
    options = vars(parser.parse_args())
    resume = options.pop("resume")
    config = TrainConfig(**{name: value for name, value in options.items() if value is not None})
    train_model(config, resume=resume)


if __name__ == "__main__":
    main()
