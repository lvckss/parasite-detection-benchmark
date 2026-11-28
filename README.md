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
| [`models/`](models/) | Checkpoints de los baselines en Git LFS y pesos preentrenados locales |
| [`results/metrics/`](results/metrics/) | Métricas de los experimentos |
| [`results/figures/`](results/figures/) | Figuras para comparar y presentar resultados |

> [!NOTE]
> Los datos, los pesos preentrenados y `.venv` se quedan en cada máquina. Git comparte el código, la configuración, los README, las métricas y figuras, y los checkpoints de los baselines mediante Git LFS.

Instala Git LFS antes de clonar para recibir los checkpoints entrenados. Si ya tienes el repo clonado, ejecuta:

```bash
git lfs install
git lfs pull
```

Se comparten `best_model.pth` y `last_checkpoint.pth` de Faster R-CNN en `models/trained/fasterrcnn_v2/baseline/`, y `best.pt` y `last.pt` de YOLO en `models/trained/yolov10s/baseline/weights/`. Los archivos auxiliares de entrenamiento y las demás ejecuciones permanecen locales. `just weights` descarga únicamente los pesos preentrenados en COCO.

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
| 7. Evaluación de YOLOv10-S en val y test | `scripts/evaluation/evaluate_yolo.py --split {val,test}` | Los mismos archivos que el paso 4 en `results/metrics/yolov10s/<run>/` |
| 8. Análisis de errores de YOLOv10-S | [`06_yolov10s_test_analysis.ipynb`](notebooks/06_yolov10s_test_analysis.ipynb) | Las mismas figuras que el paso 5 en `results/figures/yolov10s/<run>/` |
| 9. Ensemble de ambos detectores | `scripts/evaluation/evaluate_ensemble.py` y [`07_ensemble_analysis.ipynb`](notebooks/07_ensemble_analysis.ipynb) | WBF ajustado en val, métricas en `results/metrics/ensemble/<nombre>/` y figuras comparativas en `results/figures/ensemble/<nombre>/` |
| 10. Crear una variante aumentada | `scripts/create_augmented_dataset.py` | Nuevas imágenes y anotaciones solo para train; val y test se copian sin cambios |
| 11. Comparar ambos detectores por variante | [Guía de Colab](#guía-paso-a-paso-en-google-colab) | Una ejecución y un nombre de run distintos por modelo y variante |

Los detalles de cada script están en [`scripts/README.md`](scripts/README.md). Todo usa la semilla 42. Las decisiones (época, umbral de confianza) se toman en validación, y test solo se mide.

Los notebooks de entrenamiento comparten un [dashboard LIVE/REPORT](src/parasite_detection/dashboard/README.md), con métricas y gráficas en una sola salida. Cada ejecución exporta un HTML autónomo junto al CSV y conserva un resumen visual en el notebook al guardarlo.

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

Registro de las ejecuciones (rellenar solo cuando se hayan completado):

| Variante | Cambios respecto a la anterior | Ejecución Faster R-CNN | Ejecución YOLOv10-S | Val mAP COCO (FRCNN / YOLO) | Test mAP COCO (FRCNN / YOLO) | Estado |
|---|---|---|---|---:|---:|---|
| `baseline` | Train deduplicado, sin augmentación | `baseline` | `baseline` | 0,663 / 0,613 | 0,623 / — | Hecho · falta test de YOLO |
| `aug-v1` | Configuración actual de `augmentation.json` | `aug_v1` | `aug_v1` | — | — | Pendiente |
| `aug-v2` | Indicar transformaciones y parámetros añadidos | — | — | — | — | Pendiente |

### Guía paso a paso en Google Colab

Experimento `aug_v1`: **dos entrenamientos**, uno por detector, con el train aumentado. Los baselines de ambos modelos ya están entrenados y son la referencia. Todos los hiperparámetros coinciden con los de su baseline y solo cambia el dataset de train. Val y test son los mismos.

El notebook está pensado para una GPU de Colab (T4 o superior). Los datos se regeneran dentro de Colab con los mismos scripts y la semilla 42, así que no hace falta subir el dataset. Una copia periódica guarda checkpoints y métricas en Google Drive para no perderlos si se corta la sesión.

**0. Entorno de ejecución → Cambiar tipo de entorno → GPU.** Después, en celdas sucesivas:

```python
# 1. Montar Drive y definir rutas
from google.colab import drive
drive.mount("/content/drive")
REPO = "/content/parasite-detection-benchmark"
BACKUP = "/content/drive/MyDrive/parasites-runs"
```

```bash
%%bash
# 2. Clonar el repo (sin bajar los checkpoints de Git LFS, que no hacen falta para entrenar)
cd /content
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/lvckss/parasite-detection-benchmark.git
cd parasite-detection-benchmark

# 3. Instalar las versiones exactas de uv.lock (Ultralytics debe ser 8.4.215)
pip install -q uv
uv sync --locked --extra cu128
.venv/bin/python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
```

```bash
%%bash
# 4. Pesos COCO y datos: descarga → conversión → deduplicado → variante aumentada
cd /content/parasite-detection-benchmark
PY=.venv/bin/python
$PY scripts/models/download_weights.py
$PY scripts/data/download_dataset.py
$PY scripts/data/prepare_dataset.py
$PY scripts/data/deduplicate_train.py          # 1157 imágenes de train
$PY scripts/create_augmented_dataset.py \
    --source-dir data/preprocessed/parasites-1s07h-dedup \
    --output-dir data/preprocessed/parasites-1s07h-augmented-v1 \
    --config configs/augmentation.json          # 4628 imágenes de train
```

```python
# 5. Copia de seguridad en Drive cada 5 minutos, en segundo plano, mientras entrena
import subprocess, threading, time
def backup():
    while True:
        subprocess.run(f"mkdir -p {BACKUP}/models {BACKUP}/metrics && "
                       f"cp -a {REPO}/models/trained/. {BACKUP}/models/ && "
                       f"cp -a {REPO}/results/metrics/. {BACKUP}/metrics/", shell=True)
        time.sleep(300)
threading.Thread(target=backup, daemon=True).start()
```

> [!NOTE]
> No se usan enlaces simbólicos a Drive: `train_yolo.py` rechaza empezar una ejecución si su carpeta ya existe, para no mezclar experimentos.

**6. Entrenamientos.** Cada uno en su propia celda, en el mismo kernel que la celda 5 para que la copia siga activa:

```python
# a) YOLOv10-S + aug_v1 (por defecto: 100 épocas, lote 8, SGD lr 0,01, paciencia 20, como el baseline)
!cd {REPO} && .venv/bin/python scripts/training/train_yolo.py --device cuda \
    --data-dir data/preprocessed/parasites-1s07h-augmented-v1 --run-name aug_v1
```

```python
# b) Faster R-CNN v2 + aug_v1 (mismos ajustes que results/metrics/fasterrcnn_v2/baseline/settings.json)
!cd {REPO} && .venv/bin/python scripts/training/train_faster_rcnn.py --device cuda \
    --data-dir data/preprocessed/parasites-1s07h-augmented-v1 --run-name aug_v1 \
    --epochs 20 --batch-size 2 --patience 10
```

> [!WARNING]
> En `train_faster_rcnn.py`, `--data-dir` por defecto apunta al train **original** (`data/converted/...`), no al deduplicado. Indícalo siempre de forma explícita. El baseline de Faster R-CNN se entrenó con `--patience 10`; el valor por defecto actual es 5.

Si Colab corta la sesión, repetir los pasos 1–4, recuperar la copia de Drive, lanzar de nuevo la celda 5 y reanudar el entrenamiento que estuviera a medias:

```python
!cp -a {BACKUP}/models/. {REPO}/models/trained/ && cp -a {BACKUP}/metrics/. {REPO}/results/metrics/
# YOLO
!cd {REPO} && .venv/bin/python scripts/training/train_yolo.py --resume models/trained/yolov10s/aug_v1/weights/last.pt
# Faster R-CNN
!cd {REPO} && .venv/bin/python scripts/training/train_faster_rcnn.py \
    --resume models/trained/fasterrcnn_v2/aug_v1/last_checkpoint.pth \
    --data-dir data/preprocessed/parasites-1s07h-augmented-v1 --run-name aug_v1 --epochs 20 --patience 10
```

**7. Evaluar en val y test** solo al terminar y sin volver a ajustar nada mirando test. Val y test no tienen augmentación, así que basta con el dataset deduplicado por defecto:

```python
!cd {REPO} && for s in val test; do \
    .venv/bin/python scripts/evaluation/evaluate_faster_rcnn.py --run-name aug_v1 --split $s --device cuda; \
    .venv/bin/python scripts/evaluation/evaluate_yolo.py --run-name aug_v1 --split $s --device cuda; \
  done
```

**8. Llevarse los resultados.** Esperar a que la copia de la celda 5 se ejecute una última vez, o forzarla con `!cp -a {REPO}/results/metrics/. {BACKUP}/metrics/`. Pasar a `results/metrics/` del repo local los archivos ligeros (CSV, JSON y HTML) de `fasterrcnn_v2/aug_v1/` y `yolov10s/aug_v1/`, y rellenar la tabla de arriba. Los pesos (`.pt`, `.pth`) se quedan en Drive.

> [!TIP]
> Con 4628 imágenes de train, cada época dura unas 4 veces más que en el baseline. Calculo de forma aproximada varias horas por modelo en una T4. Si no hay tiempo para todo, entrena primero YOLO y deja Faster R-CNN para otra sesión. Con Colab Pro puedes elegir una GPU L4 o A100. Al comparar, ten en cuenta que en el mismo número de épocas el modelo aumentado da 4 veces más pasos de gradiente.

## Resultados

Baselines con el train deduplicado, imágenes de 640 px y semilla 42; mejor checkpoint por mAP de validación. Las métricas mAP son de `pycocotools`.

- **Faster R-CNN v2:** 20 épocas, lote 2, SGD con lr 0,0025, warmup y coseno; mejor época: 16. Entrenado con AMP en GPU.
- **YOLOv10-S:** 100 épocas previstas, 60 completadas con parada temprana tras 20 sin mejora; mejor época: 40. Lote 8, SGD con lr 0,01, warmup de 3 épocas y coseno, sin aumentos de datos. Entrenado con AMP en una NVIDIA GeForce RTX 3080.

| Modelo | Partición | mAP@[.5:.95] | mAP@0.5 | mAP@0.75 | Inferencia (CPU, lote 1, mediana) |
|---|---|---|---|---|---|
| Faster R-CNN v2 | val | 0,663 | 0,809 | 0,791 | — |
| Faster R-CNN v2 | test | 0,623 | 0,784 | 0,745 | 3241 ms/img |
| YOLOv10-S | val | 0,612 | 0,763 | 0,722 | — |
| YOLOv10-S | test | 0,537 | 0,683 | 0,638 | 265 ms/img |
| Ensemble (WBF) | val | 0,686 | 0,835 | 0,818 | — |
| Ensemble (WBF) | test | 0,635 | 0,789 | 0,756 | 3507 ms/img |

Todas las métricas se han calculado en el mismo portátil con `evaluate_faster_rcnn.py` y `evaluate_yolo.py` (`results/metrics/<modelo>/baseline/`). Ambos checkpoints se reproducen: Faster R-CNN da exactamente las métricas de su primera evaluación y YOLOv10-S da 0,612 en val frente a 0,613 al entrenar (CPU en FP32 frente a GPU).

Los tiempos salen de [`benchmark_latency.py`](scripts/evaluation/benchmark_latency.py), que alterna los dos modelos imagen a imagen sobre las mismas 50 imágenes de test en la misma CPU (Intel, 10 hilos), de modo que una ralentización de la CPU afecta a ambos por igual ([`test_cpu.json`](results/metrics/latency/test_cpu.json)). En la misma imagen, YOLOv10-S es unas 12,6 veces más rápido. Las cifras absolutas dependen de la máquina; con la CPU a pleno rendimiento fueron unos 2,3 s y 227 ms por imagen.

Una repetición del entrenamiento de Faster R-CNN con la misma configuración y semilla dio 0,657 de mAP en val frente a 0,663 ([`baseline_rerun_lucas/`](results/metrics/fasterrcnn_v2/baseline_rerun_lucas/README.md)). Diferencias de menos de unas 0,01 entre experimentos no son concluyentes.

Detalle por especie, curvas PR, análisis de errores y ejemplos en [`04_faster_rcnn_test_analysis.ipynb`](notebooks/04_faster_rcnn_test_analysis.ipynb) y [`06_yolov10s_test_analysis.ipynb`](notebooks/06_yolov10s_test_analysis.ipynb); la evolución del entrenamiento de YOLO, en [`05_yolov10s_training.ipynb`](notebooks/05_yolov10s_training.ipynb).

**Mejora: ensemble de ambos detectores.** Fusiona las predicciones de Faster R-CNN y YOLOv10-S con *Weighted Boxes Fusion* (peso de YOLO 0,75, elegido en val). En test mejora a Faster R-CNN en +0,012 de mAP (IC 95 % [+0,000, +0,026]) con un 8 % más de latencia, sobre todo porque encuentra más huevos y ajusta mejor las cajas. Dar a YOLO un peso distinto según el tipo de imagen (clasificador entrenado con nuestras etiquetas manuales) no aportó. Decisiones y justificación en [`docs/decisiones-mejora-ensemble.md`](docs/decisiones-mejora-ensemble.md); resultados en [`07_ensemble_analysis.ipynb`](notebooks/07_ensemble_analysis.ipynb). Las demás estrategias de mejora están en curso.

## Estado del proyecto

Referencia: [guion de la práctica](docs/guion_practica_deteccion.pdf) y [notas de los profesores](docs/notas-profesores.md). Entrega: ZIP con diapositivas en PDF, código reproducible con semillas y este README.

### Hecho

- Entorno reproducible con `uv`, descarga fijada del dataset y conversión a COCO/YOLO.
- EDA y deduplicado del train frente a val/test (1484 → 1157 imágenes) para evitar fuga de datos.
- Faster R-CNN v2: entrenamiento completo (todas las capas), validación con `val_loss`, parada temprana, evaluación en val/test y análisis de errores con figuras. **Baseline terminado.**
- YOLOv10-S: entrenador con el motor oficial de Ultralytics, evaluación COCO común con Faster R-CNN (`evaluate_yolo.py`), evaluación en val/test y análisis de errores con figuras. **Baseline terminado** (test mAP 0,537).
- Ambos baselines evaluados en la misma máquina y latencias comparadas con `benchmark_latency.py` (YOLOv10-S unas 12,6 veces más rápido).
- Checkpoints de ambos baselines compartidos con Git LFS.
- Generador de datasets aumentados (`configs/augmentation.json`, inspirado en el paper de `docs/`), visor web de augmentaciones y etiquetador visual.
- Dashboard de entrenamiento compartido por ambos modelos.
- Mejora por ensemble (WBF) de ambos detectores, con comprobación de la ponderación por tipo de imagen: test mAP 0,635 frente a 0,623 de Faster R-CNN ([decisiones](docs/decisiones-mejora-ensemble.md)).

### Pendiente

- [ ] **`aug_v1` en ambos detectores** y comparación con su baseline ([guía de Colab](#guía-paso-a-paso-en-google-colab)).
- [ ] Al menos una o dos estrategias de mejora más allá de la augmentación (ver ideas).
- [ ] Tabla y gráficas finales baseline vs. mejoras (mAP@0.5, mAP@[.5:.95], latencia).
- [ ] Diapositivas (11–12 min, intervienen los tres) y ZIP `GRUPO_XX.zip`.

### Ideas de mejora

Ordenadas de menor a mayor esfuerzo. Cada una debe justificarse con una hipótesis (la originalidad pesa un 30 % de la nota).

| Idea | Hipótesis / justificación | Esfuerzo |
|---|---|---|
| Ajuste de umbral de confianza y NMS en val | Las predicciones ya están guardadas en `*_predictions.json`; no requiere reentrenar | Bajo |
| Test-time augmentation (flips) | Los huevos no tienen orientación canónica, así que los flips no cambian la clase | Bajo |
| Ensemble Faster R-CNN + YOLO con *Weighted Boxes Fusion* | Un two-stage y un one-stage cometen errores distintos | Medio |
| `aug_v2`: activar `hue_saturation`/`channel_shuffle` | Las tinciones y el microscopio cambian el color entre imágenes | Medio |
| Pérdida ponderada por clase o muestreo balanceado | Compensa el desequilibrio entre especies, si el EDA lo confirma | Medio |
| Congelar capas del backbone vs. entrenar todo; LR distinto backbone/cabeza | Estrategia de *transfer learning* sugerida en el guion | Medio |
| SGD vs. AdamW | Comparación de optimizadores sugerida en el guion | Medio |
| Mayor resolución de entrada (p. ej. 800–1024 px) | Los huevos más pequeños podrían perder detalle a 640 px (comprobar en el EDA) | Medio-alto |

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
