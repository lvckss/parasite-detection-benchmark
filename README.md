# Parasite detection benchmark

Comparison of Faster R-CNN v2 and YOLOv10-S for parasite detection using the parasites-1s07h dataset.

Práctica experimental de Visión por Computador. Modelos asignados:

- **Faster R-CNN v2:** `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`, preentrenado en COCO.
- **YOLOv10-S:** `ultralytics.YOLO("yolov10s.pt")`, preentrenado en COCO.

## Inicio rápido

Instala [uv](https://docs.astral.sh/uv/getting-started/installation/). El descargador obtiene los datos originales desde Hugging Face.

```bash
git clone https://github.com/lvckss/parasite-detection-benchmark.git
cd parasite-detection-benchmark

# Entorno para explorar los datos o trabajar sin una GPU NVIDIA.
./scripts/setup_env.sh cpu

# Descarga los datos y prepara los dos formatos de anotaciones.
python3 scripts/download_dataset.py
.venv/bin/python scripts/prepare_dataset.py

# Abre Jupyter y selecciona el kernel VISION P5.
.venv/bin/jupyter lab
```

Para una GPU NVIDIA compatible con CUDA 12.8, utiliza `./scripts/setup_env.sh cu128` en lugar de `cpu`. El driver debe ser compatible; compruébalo con `nvidia-smi`. Los backends son alternativos y no deben instalarse juntos.

El script ejecuta `uv sync --locked --extra cpu` o `uv sync --locked --extra cu128`. `uv.lock` fija las versiones compartidas; cada compañero crea su propia `.venv`. Para ejecutar comandos después, utiliza `.venv/bin/python` o `uv run --no-sync`, conservando el backend instalado. En Windows puedes usar `uv sync --locked --extra cpu` y `.venv\Scripts\python.exe` directamente.

El kernel se registra dentro del entorno. Para seleccionarlo desde un Jupyter instalado fuera de esta `.venv`, regístralo también en tu usuario:

```bash
.venv/bin/python -m ipykernel install --user --name vision-p5 --display-name "VISION P5"
```

## Organización

```text
.agents/skills/             Skills compartidas y descubiertas por Codex
.claude/skills/             Adaptadores para Claude Code
AGENTS.md / CLAUDE.md       Instrucciones comunes de trabajo
configs/                   Configuración del dataset y futuros experimentos
scripts/                   Creación del entorno, descarga y preparación
src/parasite_detection/    Helpers Python compartidos por los notebooks
notebooks/                 Exploración y futuros entrenamientos
data/raw/                  Parquet originales, descargados localmente
data/converted/            Imágenes extraídas y anotaciones COCO/YOLO
data/preprocessed/         Reservada para futuros preprocesamientos del equipo
models/                    Pesos descargados y checkpoints locales
results/metrics/           Métricas pequeñas que queramos compartir
results/figures/           Figuras para el análisis y la presentación
```

Los datos descargados y generados, `.venv`, los pesos y las credenciales están excluidos de nuevos commits. Los README de las carpetas de datos sí se versionan. El descargador reutiliza los Parquet descargados si están completos y coinciden con los checksums.

Las fases se explican en [data/README.md](data/README.md). `converted/` cambia la organización y el formato de las anotaciones, conservando los bytes originales de las imágenes. `preprocessed/` contiene únicamente su README; el equipo decidirá e implementará ahí sus futuras variantes de preprocesamiento.

## Datos y etiquetas

Fuente: [Francesco/parasites-1s07h](https://huggingface.co/datasets/Francesco/parasites-1s07h), procedente de Roboflow-100. La revisión y los checksums están fijados en `configs/dataset.json`. La descarga es pública y no requiere una API key de Roboflow.

Se mantienen las particiones **publicadas como Parquet**, sin repartir de nuevo ni aumentar imágenes:

| Hugging Face | Carpeta preparada | Imágenes |
|---|---|---:|
| train | train | 1484 |
| validation | val | 215 |
| test | test | 411 |

**Diferencia entre formatos de origen:** el `dataset.tar.gz` del mismo repositorio contiene `train=1484`, `valid=411` y `test=215`. Se ha comprobado mediante hashes de las imágenes que `test` del tar coincide con `validation` de los Parquet, y `valid` del tar coincide con `test` de los Parquet. Este proyecto usa los splits Parquet, coherentes con `dataset_info.json`, de forma explícita para que todos trabajemos con la misma partición. No mezcles ambos formatos por el nombre de sus carpetas.

La preparación genera una única copia de las imágenes para ambos modelos:

```text
data/converted/parasites-1s07h/
├── images/{train,val,test}/
├── annotations/instances_{train,val,test}.json
├── labels/{train,val,test}/
├── dataset.yaml
├── class_mapping.json
└── summary.json
```

**Faster R-CNN:** las anotaciones COCO conservan los IDs 1–8; la etiqueta 0 queda reservada a background en torchvision. La cabeza del detector tendrá 9 salidas: 8 especies y background. Las cajas de COCO están en `[x, y, width, height]`; al construir el dataset de torchvision hay que convertirlas a `[x1, y1, x2, y2]`.

**YOLOv10-S:** `dataset.yaml` apunta a las mismas imágenes y las etiquetas usan clases 0–7 y cajas normalizadas `[cx, cy, width, height]`. `class_mapping.json` documenta la correspondencia. La categoría general `parasites` que aparece con ID 0 en los metadatos originales no se trata como una novena especie.

Puedes volver a ejecutar los dos scripts: se reutilizan las descargas completas y se regeneran las anotaciones. Si mueves el repo, repite la preparación para actualizar la ruta local de `dataset.yaml`.

El notebook `notebooks/01_dataset_overview.ipynb` muestra los recuentos, clases y ejemplos con sus cajas. Los entrenamientos y sus resultados se añadirán durante la práctica.

## Trabajo en equipo

Las instrucciones compartidas están en `AGENTS.md`; Claude Code las importa desde `CLAUDE.md`. La skill `semantic-commits-and-push` se invoca como `$semantic-commits-and-push` en Codex y `/semantic-commits-and-push` en Claude Code.

Guarda la configuración de cada experimento junto a sus métricas y figuras. Usa validación para elegir los cambios y reserva test para la comparación final. Mantén los pesos en `models/` y los logs grandes fuera de Git.
