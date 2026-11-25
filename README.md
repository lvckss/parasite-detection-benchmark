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

## Webs de etiquetado y revisión visual

Ejecuta los comandos desde la raíz del repositorio con el entorno virtual activo. Si aún no tienes el train deduplicado y su orden visual, prepara los datos y genera el manifiesto:

```bash
just dataset
just prepare
python scripts/data/deduplicate_train.py
python scripts/labeling/order_train_images_by_similarity.py --device cuda
```

Si no tienes GPU NVIDIA, cambia `--device cuda` por `--device cpu`. La ordenación se necesita una sola vez, salvo que cambie el conjunto deduplicado.

### Etiquetar imágenes del train

```bash
just label
```

Esto inicia `scripts/labeling/classify_train_images.py`, abre la web local en `http://127.0.0.1:8765` y guarda las etiquetas en `results/visual-labels/train_dedup_labels.json`. Detén el servidor con **Ctrl+C**. También puedes ejecutarlo directamente con `python scripts/labeling/classify_train_images.py`; usa `--port` para elegir otro puerto y `--no-browser` para no abrir el navegador automáticamente.

### Revisar originales y augmentaciones

Genera primero una variante aumentada —por defecto, desde el train deduplicado— y luego arranca el comparador:

```bash
python scripts/create_augmented_dataset.py
python scripts/labeling/compare_train_augmentations.py --data-dir data/preprocessed/parasites-1s07h-augmented-v1
```

La web local se abre en `http://127.0.0.1:8766`. Permite comparar cada original con sus variantes y ver qué transformaciones se aplicaron. Si ya clasificaste las imágenes, el comparador también puede filtrar por esas etiquetas; no es obligatorio ejecutar primero la web de etiquetado. El dataset debe incluir metadatos de transformaciones, que genera automáticamente la versión actual del script. Detén el servidor con **Ctrl+C**; puedes cambiar el puerto con `--port` o evitar la apertura automática con `--no-browser`.

Para inspeccionar un dataset antiguo que no guarde `applied_transforms`, añade esos metadatos indicando el mismo origen, salida y configuración con que se creó (esto no modifica las imágenes):

```bash
python scripts/create_augmented_dataset.py --source-dir data/converted/parasites-1s07h --output-dir data/preprocessed/parasites-1s07h-augmented --write-transform-metadata
python scripts/labeling/compare_train_augmentations.py --data-dir data/preprocessed/parasites-1s07h-augmented
```

## Flujo experimental

| Paso | Dónde | Qué produce |
|---|---|---|
| 1. Exploración del dataset | [`eda_preliminar.ipynb`](notebooks/eda_preliminar.ipynb) | Distribución de clases, tamaños, color y casi duplicados entre particiones |
| 2. Comprobación de datos para Faster R-CNN | [`02_faster_rcnn_data_check.ipynb`](notebooks/02_faster_rcnn_data_check.ipynb) | Verificación de cajas, tensores y `DataLoader` |
| 3. Entrenamiento de Faster R-CNN v2 | [`03_faster_rcnn_training.ipynb`](notebooks/03_faster_rcnn_training.ipynb) o `scripts/training/train_faster_rcnn.py` | Checkpoints en `models/trained/fasterrcnn_v2/<run>/`, `training.csv` y `settings.json` |
| 4. Evaluación en val y test | `scripts/evaluation/evaluate_faster_rcnn.py --split {val,test}` | Métricas COCO, AP por clase, curvas PR, latencia y predicciones en `results/metrics/fasterrcnn_v2/<run>/` |
| 5. Análisis de errores y figuras | [`04_faster_rcnn_test_analysis.ipynb`](notebooks/04_faster_rcnn_test_analysis.ipynb) | Curvas, matriz de confusión, desglose tipo TIDE y ejemplos en `results/figures/fasterrcnn_v2/<run>/` |
| 6. Entrenamiento de YOLOv10-S | `scripts/training/train_yolo.py` | Checkpoints en `models/trained/yolov10s/<run>/` y métricas en `results/metrics/yolov10s/<run>/` |
| 7. Crear una variante aumentada | `scripts/create_augmented_dataset.py` | Nuevas imágenes y anotaciones solo para train; val y test se copian sin cambios |
| 8. Comparar ambos detectores por variante | Notebooks de entrenamiento en Colab/Kaggle | Una ejecución y un nombre de run distintos por modelo y variante |

Los detalles de cada script están en [`scripts/README.md`](scripts/README.md). Todo usa la semilla 42. Las decisiones (época, umbral de confianza) se toman en validación, y test solo se mide.

Los notebooks de entrenamiento comparten un [dashboard LIVE/REPORT](src/parasite_detection/dashboard/README.md), con métricas y gráficas en una sola salida. Cada ejecución exporta un HTML autónomo junto al CSV y conserva un resumen visual en el notebook al guardarlo. La [demo con métricas simuladas](notebooks/05_training_dashboard_demo.ipynb) permite probarlo sin entrenar.

## Experimentos de aumentación en Colab/Kaggle

El siguiente bloque de trabajo es medir cómo cambia el rendimiento de **Faster R-CNN v2** y **YOLOv10-S** al ampliar progresivamente las augmentaciones. Es un plan experimental: las ejecuciones de esta sección aún no se consideran resultados hasta que sus métricas se registren.

### Preparar variantes

Partir del train deduplicado evita volver a introducir en train imágenes casi duplicadas de val o test. El generador conserva los originales, aplica las transformaciones únicamente a train y copia val/test sin modificar. Cada configuración debe tener su propio directorio de salida; el generador no sobrescribe uno existente.

La configuración actual de [`configs/augmentation.json`](configs/augmentation.json) crea tres variantes por imagen de train, además de conservar cada original. Si se aplica al train deduplicado de 1.157 imágenes, el resultado esperado es 4.628 imágenes de train. La configuración actual activa flips horizontal y vertical, rotación, escala, cambios de brillo y contraste y desenfoque gaussiano. Las demás transformaciones están desactivadas. El directorio antiguo `parasites-1s07h-augmented/` se generó desde el train completo de 1.484 imágenes; para mantener la evaluación sin duplicados, genera una variante nueva desde el train deduplicado.

```bash
python scripts/create_augmented_dataset.py \
    --source-dir data/preprocessed/parasites-1s07h-dedup \
    --output-dir data/preprocessed/parasites-1s07h-augmented-v1 \
    --config configs/augmentation.json
```

Para cada variante posterior, guardar una configuración identificable y usar otra ruta de salida, por ejemplo `parasites-1s07h-augmented-v2`. Añadir una transformación o un grupo cada vez facilita relacionar los cambios con las métricas; mantener iguales las transformaciones ya probadas y documentar las probabilidades y rangos usados. La carpeta de datos generada no se sube a Git.

### Entrenar en la nube y comparar

1. Subir al almacenamiento privado de Colab o Kaggle el código/configuración del experimento y la variante de datos preparada. Descargar o montar los pesos preentrenados por separado; no guardar datasets ni checkpoints en el repositorio.
2. Seleccionar la variante antes de importar los módulos en el notebook, por ejemplo con `PARASITE_DATA_DIR=/ruta/al/dataset`; usar `PARASITE_RUN_NAME=aug_v1` para identificar la ejecución. Ambos notebooks consultan estas variables y, si no están definidas, usan el train deduplicado y el nombre del baseline. En los scripts de terminal también se pueden indicar `--data-dir` y `--run-name`. Usar los mismos splits, pesos iniciales, semilla e hiperparámetros del baseline correspondiente. Cambiar solo la variante de augmentación que se está evaluando.
3. Entrenar los dos modelos con nombres de ejecución únicos, por ejemplo `fasterrcnn_aug_v1` y `yolov10s_aug_v1` (o `aug_v1` en `PARASITE_RUN_NAME`; cada modelo guarda en su propio directorio). Conservar por separado checkpoints, `settings.json`, CSV de entrenamiento y métricas; anotar también el commit del código y la configuración utilizada.
4. Usar **validation** para elegir época/checkpoint y tomar decisiones. Evaluar **test** solo después de cerrar esas decisiones, con las imágenes y anotaciones originales sin augmentar. No ajustar configuraciones mirando resultados de test.
5. Repetir el proceso con la siguiente variante y comparar cada detector con su propio baseline y sus ejecuciones anteriores. Registrar mAP COCO@[.5:.95] como métrica principal, además de mAP50, AP por clase y, si se compara latencia, el hardware usado.

> [!IMPORTANT]
> Ambos entrenadores validan la consistencia de las anotaciones e imágenes. YOLO acepta el baseline del manifiesto o una variante generada que conserva la correspondencia de las imágenes fuente y `augmentation_config.json`; también comprueba que las cajas COCO y YOLO coincidan. Las transformaciones de Ultralytics en tiempo de entrenamiento siguen desactivadas: las augmentaciones de estas pruebas son las materializadas en el dataset.

Registro de las ejecuciones de augmentación (rellenar solo cuando se hayan completado):

| Variante | Cambios respecto a la anterior | Ejecución Faster R-CNN | Ejecución YOLOv10-S | Val mAP COCO | Test mAP COCO | Estado |
|---|---|---|---|---:|---:|---|
| `aug-v1` | Configuración actual de `augmentation.json` | — | — | — | — | Pendiente |
| `aug-v2` | Indicar transformaciones y parámetros añadidos | — | — | — | — | Pendiente |

## Resultados

Baseline con el train deduplicado: 20 épocas, SGD con lr 0,0025, warmup y coseno, imágenes de 640 px; mejor época por mAP de validación. Las métricas son de `pycocotools`.

| Modelo | Partición | mAP@[.5:.95] | mAP@0.5 | mAP@0.75 | Inferencia |
|---|---|---|---|---|---|
| Faster R-CNN v2 | val | 0,663 | 0,809 | 0,791 | — |
| Faster R-CNN v2 | test | 0,623 | 0,784 | 0,745 | 2,1 s/img (CPU, lote 1) |
| YOLOv10-S | — | en curso | | | |

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
