# Parasite Detection Benchmark

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/dependencies-uv-DE5FE9?logo=uv&logoColor=white)
![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white)

> [!NOTE]
> **Uso de herramientas de IA.** Codex y Claude Code se utilizan como apoyo para organizar el repositorio, configurar el entorno y mantener scripts auxiliares de descarga y conversión de datos. El equipo diseña e implementa manualmente la lógica de preprocesamiento y el pipeline experimental, con razonamiento propio y decisiones justificadas a partir de las hipótesis planteadas y los resultados obtenidos.

**Faster R-CNN v2 vs. YOLOv10-S** sobre imágenes de microscopía de ocho especies de parásitos. Práctica de Visión por Computador I.

[Empezar](#empezar) · [Dataset](data/README.md) · [Exploración](notebooks/eda_preliminar.ipynb) · [Resultados](#resultados) · [Equipo](#equipo)

| Detector | Arquitectura | Implementación |
|---|---|---|
| Faster R-CNN v2 | Two-stage · ResNet-50 + FPN | `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2` |
| YOLOv10-S | One-stage · variante Small | `ultralytics.YOLO("models/pretrained/yolov10s.pt")` |

Ambos parten de pesos preentrenados en COCO. El dataset es [Francesco/parasites-1s07h](https://huggingface.co/datasets/Francesco/parasites-1s07h), de Roboflow-100.

## Empezar

Necesitas [uv](https://docs.astral.sh/uv/getting-started/installation/). Desde una terminal:

```bash
git clone https://github.com/lvckss/parasite-detection-benchmark.git
cd parasite-detection-benchmark

./scripts/environment/setup_env.sh cpu
source .venv/bin/activate

# Descargar los pesos y preparar el dataset
just weights
just dataset
just prepare

# Quitar del train los casi duplicados de val/test (variante con la que se entrenan los baselines)
python scripts/data/deduplicate_train.py

# Abrir los notebooks
jupyter lab
```

`just weights` descarga los pesos COCO de Faster R-CNN v2 y YOLOv10-S indicados en el guion de la práctica. Para descargar solo uno, usa `python scripts/models/download_weights.py --only fasterrcnn` o `--only yolo`; añade `--force` para reemplazar un archivo existente. El peso de Faster R-CNN se valida con el prefijo SHA-256 publicado en su nombre; la release de YOLO no incluye una suma de referencia en el guion.

Con Jupyter abierto, selecciona el kernel **VISION P5** y sigue los notebooks de `notebooks/` para explorar el dataset y ejecutar los experimentos.

> [!TIP]
> **¿Vas a usar una GPU NVIDIA?** Cambia `cpu` por `cu128` al crear el entorno. Necesitas un driver compatible con CUDA 12.8; puedes consultarlo con `nvidia-smi`. Elige un backend por entorno.

### Configuración del entorno

El script utiliza `uv sync --locked --extra cpu` o `uv sync --locked --extra cu128`. Las versiones están en `uv.lock`; cada persona crea su propia `.venv`.

Activa el entorno con `source .venv/bin/activate` al abrir una terminal nueva en el repo. Después puedes usar `python` y `jupyter` directamente.

#### Windows (PowerShell)

```powershell
uv sync --no-editable --locked --extra cpu
.\.venv\Scripts\Activate.ps1
```

#### Jupyter externo

Con el entorno activado, registra el kernel en tu usuario si vas a abrir los notebooks desde un Jupyter instalado fuera de este entorno:

```bash
python -m ipykernel install --user --name vision-p5 --display-name "VISION P5"
```

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| [`configs/`](configs/) | Fuente del dataset y configuración de los experimentos |
| [`scripts/`](scripts/README.md) | Guía de scripts por categoría |
| [`src/parasite_detection/`](src/parasite_detection/) | Código compartido entre notebooks |
| [`notebooks/`](notebooks/) | Exploración y entrenamientos |
| [`data/raw/`](data/raw/README.md) | Parquet originales |
| [`data/converted/`](data/converted/README.md) | Imágenes extraídas y anotaciones COCO/YOLO |
| [`data/preprocessed/`](data/preprocessed/README.md) | Nuestras variantes, como train sin casi duplicados |
| [`models/`](models/) | Pesos y checkpoints locales |
| [`results/metrics/`](results/metrics/) | Métricas de los experimentos |
| [`results/figures/`](results/figures/) | Figuras para comparar y presentar resultados |

> [!NOTE]
> Los datos, los pesos y `.venv` se quedan en cada máquina. Git comparte el código, la configuración, los README y las métricas y figuras que añadamos.

## Dataset

Usamos los **splits Parquet de Hugging Face**: 1484 imágenes de entrenamiento, 215 de validación y 411 de test. La descarga es pública y tiene revisión y checksums fijados en [`configs/dataset.json`](configs/dataset.json).

La conversión conserva las imágenes originales y genera ambos formatos de anotaciones. Para entrenar usamos una variante con el **train deduplicado**: quitamos las 327 imágenes de train casi idénticas a alguna de val o test (correlación ≥ 0,90 de miniaturas en gris), así que quedan 1157 imágenes y 3015 objetos. Val y test no cambian, de modo que las métricas de test no están infladas por fuga de datos.

- [Fases, particiones y diferencia con el tar original](data/README.md)
- [Formatos de cajas y correspondencia de clases](data/converted/README.md)
- [Criterio de deduplicación](data/preprocessed/README.md)

## Flujo experimental

| Paso | Dónde | Qué produce |
|---|---|---|
| 1. Exploración del dataset | [`eda_preliminar.ipynb`](notebooks/eda_preliminar.ipynb) | Distribución de clases, tamaños, color y casi duplicados entre particiones |
| 2. Comprobación de datos para Faster R-CNN | [`02_faster_rcnn_data_check.ipynb`](notebooks/02_faster_rcnn_data_check.ipynb) | Verificación de cajas, tensores y `DataLoader` |
| 3. Entrenamiento de Faster R-CNN v2 | [`03_faster_rcnn_training.ipynb`](notebooks/03_faster_rcnn_training.ipynb) o `scripts/training/train_faster_rcnn.py` | Checkpoints en `models/trained/fasterrcnn_v2/<run>/`, `training.csv` y `settings.json` |
| 4. Evaluación en val y test | `scripts/evaluation/evaluate_faster_rcnn.py --split {val,test}` | Métricas COCO, AP por clase, curvas PR, latencia y predicciones en `results/metrics/fasterrcnn_v2/<run>/` |
| 5. Análisis de errores y figuras | [`04_faster_rcnn_test_analysis.ipynb`](notebooks/04_faster_rcnn_test_analysis.ipynb) | Curvas, matriz de confusión, desglose tipo TIDE y ejemplos en `results/figures/fasterrcnn_v2/<run>/` |
| 6. Entrenamiento de YOLOv10-S | `scripts/training/train_yolo.py` | Checkpoints en `models/trained/yolov10s/<run>/` y métricas en `results/metrics/yolov10s/<run>/` |
| 7. Evaluación de YOLOv10-S en val y test | `scripts/evaluation/evaluate_yolo.py --split {val,test}` | Los mismos archivos que el paso 4 en `results/metrics/yolov10s/<run>/` |
| 8. Análisis de errores de YOLOv10-S | [`05_yolov10s_test_analysis.ipynb`](notebooks/05_yolov10s_test_analysis.ipynb) | Las mismas figuras que el paso 5 en `results/figures/yolov10s/<run>/` |

Los detalles de cada script están en [`scripts/README.md`](scripts/README.md). Todo usa la semilla 42. Las decisiones (época, umbral de confianza) se toman en validación, y test solo se mide.

Los notebooks de entrenamiento comparten un [dashboard LIVE/REPORT](src/parasite_detection/dashboard/README.md), con métricas y gráficas en una sola salida. Cada ejecución exporta un HTML autónomo junto al CSV y conserva un resumen visual en el notebook al guardarlo. La [demo con métricas simuladas](notebooks/05_training_dashboard_demo.ipynb) permite probarlo sin entrenar.

## Resultados

Baselines con el train deduplicado, imágenes de 640 px y semilla 42; mejor checkpoint por mAP de validación. Las métricas mAP son de `pycocotools`.

- **Faster R-CNN v2:** 20 épocas, lote 2, SGD con lr 0,0025, warmup y coseno.
- **YOLOv10-S:** 100 épocas previstas, 60 completadas con parada temprana tras 20 sin mejora; mejor época: 40. Lote 8, SGD con lr 0,01, warmup de 3 épocas y coseno, sin aumentos de datos. Entrenado con AMP en una NVIDIA GeForce RTX 3080.

| Modelo | Partición | mAP@[.5:.95] | mAP@0.5 | mAP@0.75 | Inferencia |
|---|---|---|---|---|---|
| Faster R-CNN v2 | val | 0,663 | 0,809 | 0,791 | — |
| Faster R-CNN v2 | test | 0,623 | 0,784 | 0,745 | 2,1 s/img (CPU, lote 1) |
| YOLOv10-S | val | 0,613 | 0,764 | — | 21,4 ms/img (RTX 3080, validación Ultralytics) |
| YOLOv10-S | test | pendiente | — | — | — |

Los tiempos de inferencia se han medido con hardware y protocolos distintos (CPU para Faster R-CNN y GPU para YOLO), por lo que no son directamente comparables. El mAP@0.75 de YOLO no está registrado en el resumen disponible; su evaluación en test está pendiente.

Los resultados de YOLO están en [`best_validation.json`](results/metrics/yolov10s/baseline/best_validation.json), con la configuración en [`settings.json`](results/metrics/yolov10s/baseline/settings.json), la evolución por época en [`training.csv`](results/metrics/yolov10s/baseline/training.csv) y el informe en [`04_yolov10s_training.ipynb`](notebooks/04_yolov10s_training.ipynb).

Detalle por especie, curvas PR, análisis de errores y ejemplos en [`04_faster_rcnn_test_analysis.ipynb`](notebooks/04_faster_rcnn_test_analysis.ipynb). Las estrategias de mejora están pendientes.

## Equipo

| Integrante | Correo |
|---|---|
| Rodrigo Boquete Ríos | [rodrigo.boquete@rai.usc.gal](mailto:rodrigo.boquete@rai.usc.gal) |
| Alejandro Barros Monroy | [alejandro.barros0@rai.usc.gal](mailto:alejandro.barros0@rai.usc.gal) |
| Lucas García Ruiz | [lucas.garcia0@rai.usc.es](mailto:lucas.garcia0@rai.usc.es) |

### Trabajo con agentes

Codex y Claude Code comparten las reglas de [`AGENTS.md`](AGENTS.md). [`CLAUDE.md`](CLAUDE.md) las importa y las skills de `.claude/` remiten a las de `.agents/`.

| Acción | Codex | Claude Code |
|---|---|---|
| Preparar commits, subir la rama y abrir PR | `$semantic-commits-and-push` | `/semantic-commits-and-push` |
