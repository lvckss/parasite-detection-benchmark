# Parasite Detection Benchmark

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/dependencies-uv-DE5FE9?logo=uv&logoColor=white)
![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white)

**Faster R-CNN v2 vs. YOLOv10-S** sobre imágenes de microscopía de ocho especies de parásitos. Práctica de Visión por Computador.

[Empezar](#empezar) · [Dataset](data/README.md) · [Exploración](notebooks/01_dataset_overview.ipynb) · [Equipo](#equipo)

| Detector | Arquitectura | Implementación |
|---|---|---|
| Faster R-CNN v2 | Two-stage · ResNet-50 + FPN | `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2` |
| YOLOv10-S | One-stage · variante Small | `ultralytics.YOLO("yolov10s.pt")` |

Ambos parten de pesos preentrenados en COCO. El dataset es [Francesco/parasites-1s07h](https://huggingface.co/datasets/Francesco/parasites-1s07h), de Roboflow-100.

## Empezar

Necesitas [uv](https://docs.astral.sh/uv/getting-started/installation/). Desde una terminal:

```bash
git clone https://github.com/lvckss/parasite-detection-benchmark.git
cd parasite-detection-benchmark

# 1. Crear el entorno y el kernel VISION P5
./scripts/setup_env.sh cpu
source .venv/bin/activate

# 2. Descargar y convertir el dataset
python scripts/download_dataset.py
python scripts/prepare_dataset.py

# 3. Abrir Jupyter
jupyter lab
```

Selecciona **VISION P5** y abre [`01_dataset_overview.ipynb`](notebooks/01_dataset_overview.ipynb): recuentos, distribución de clases y ejemplos con sus cajas.

> [!TIP]
> **¿Vas a usar una GPU NVIDIA?** Cambia `cpu` por `cu128` al crear el entorno. Necesitas un driver compatible con CUDA 12.8; puedes consultarlo con `nvidia-smi`. Elige un backend por entorno.

<details>
<summary>Comandos uv, Windows y Jupyter externo</summary>

El script utiliza `uv sync --locked --extra cpu` o `uv sync --locked --extra cu128`. Las versiones están en `uv.lock`; cada persona crea su propia `.venv`.

Activa el entorno con `source .venv/bin/activate` al abrir una terminal nueva en el repo. Después puedes usar `python` y `jupyter` directamente.

En Windows, desde PowerShell:

```powershell
uv sync --locked --extra cpu
.\.venv\Scripts\Activate.ps1
```

Si abres los notebooks desde un Jupyter instalado fuera de este entorno, registra también el kernel en tu usuario:

```bash
python -m ipykernel install --user --name vision-p5 --display-name "VISION P5"
```

</details>

## Dónde va cada cosa

| Ruta | Contenido |
|---|---|
| [`configs/`](configs/) | Fuente del dataset y configuración de los experimentos |
| [`scripts/`](scripts/) | Entorno, descarga y conversión |
| [`src/parasite_detection/`](src/parasite_detection/) | Código compartido entre notebooks |
| [`notebooks/`](notebooks/) | Exploración y entrenamientos |
| [`data/raw/`](data/raw/README.md) | Parquet originales |
| [`data/converted/`](data/converted/README.md) | Imágenes extraídas y anotaciones COCO/YOLO |
| [`data/preprocessed/`](data/preprocessed/README.md) | Reservado para nuestros preprocesamientos |
| [`models/`](models/) | Pesos y checkpoints locales |
| [`results/metrics/`](results/metrics/) | Métricas de los experimentos |
| [`results/figures/`](results/figures/) | Figuras para comparar y presentar resultados |

> [!NOTE]
> Los datos, los pesos y `.venv` se quedan en cada máquina. Git comparte el código, la configuración, los README y las métricas y figuras que añadamos.

## Dataset

Usamos los **splits Parquet de Hugging Face**: 1484 imágenes de entrenamiento, 215 de validación y 411 de test. La descarga es pública y tiene revisión y checksums fijados en [`configs/dataset.json`](configs/dataset.json).

La conversión conserva las imágenes originales y genera ambos formatos de anotaciones. El preprocesamiento experimental lo haremos después.

- [Fases, particiones y diferencia con el tar original](data/README.md)
- [Formatos de cajas y correspondencia de clases](data/converted/README.md)

## Equipo

Codex y Claude Code comparten las reglas de [`AGENTS.md`](AGENTS.md). [`CLAUDE.md`](CLAUDE.md) las importa y las skills de `.claude/` remiten a las de `.agents/`.

| Acción | Codex | Claude Code |
|---|---|---|
| Preparar commits, subir la rama y abrir PR | `$semantic-commits-and-push` | `/semantic-commits-and-push` |

Cada experimento debe dejar su configuración, métricas y figuras. Elegimos las mejoras con validación y reservamos test para la comparación final.
