# Parasite Detection Benchmark

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/dependencies-uv-DE5FE9?logo=uv&logoColor=white)
![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white)

> [!NOTE]
> **Uso de herramientas de IA.** Codex y Claude Code se utilizan como apoyo para organizar el repositorio, configurar el entorno y mantener scripts auxiliares de descarga y conversión de datos. El equipo diseña e implementa manualmente la lógica de preprocesamiento y el pipeline experimental, con razonamiento propio y decisiones justificadas a partir de las hipótesis planteadas y los resultados obtenidos.

**Faster R-CNN v2 vs. YOLOv10-S** sobre imágenes de microscopía de ocho especies de parásitos. Práctica de Visión por Computador I.

[Empezar](#empezar) · [Dataset](data/README.md) · [Exploración](notebooks/01_dataset_overview.ipynb) · [Equipo](#equipo)

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

La conversión conserva las imágenes originales y genera ambos formatos de anotaciones. El preprocesamiento experimental lo haremos después.

- [Fases, particiones y diferencia con el tar original](data/README.md)
- [Formatos de cajas y correspondencia de clases](data/converted/README.md)

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
