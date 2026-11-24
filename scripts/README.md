# Scripts

Herramientas del proyecto, agrupadas por tarea. Ejecuta los comandos desde la raíz del repositorio, con el entorno virtual activo cuando corresponda.

```text
scripts/
├── data/
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   └── deduplicate_train.py
├── environment/
│   └── setup_env.sh
├── models/
│   └── download_weights.py
├── labeling/
│   ├── order_train_images_by_similarity.py
│   ├── classify_train_images.py
│   ├── compare_train_augmentations.py
│   ├── visual_labeler/
│   └── augmentation_viewer/
└── training/
    └── train_faster_rcnn.py
```

## Entorno y pesos de los modelos

| Archivo | Función | Uso |
|---|---|---|
| [`setup_env.sh`](environment/setup_env.sh) | Crea o actualiza `.venv`, instala las dependencias y registra el kernel de Jupyter. Acepta `cpu` o `cu128`. | `bash scripts/environment/setup_env.sh cu128` |
| [`download_weights.py`](models/download_weights.py) | Descarga a `models/pretrained/` los pesos COCO de Faster R-CNN v2 y YOLOv10-S. Omite los archivos existentes y mueve los de la estructura anterior a esta carpeta; `--force` los reemplaza y `--only fasterrcnn` o `--only yolo` descarga un solo modelo. | `just weights` |

Los pesos y archivos descargados se mantienen localmente y no se incluyen en Git.

## Entrenamiento de Faster R-CNN

[`training/train_faster_rcnn.py`](training/train_faster_rcnn.py) entrena Faster R-CNN v2 con ResNet-50 FPN, valida al final de cada época y guarda `train_loss`, `val_loss`, mAP COCO y mAP50. Parte de los pesos COCO descargados y ajusta todas las capas al dataset. Por defecto usa el train original y selecciona CUDA si está disponible.

```bash
python scripts/training/train_faster_rcnn.py --epochs 20 --batch-size 2
```

Con `--patience 0` se completan todas las épocas pedidas. Para entrenar con la variante deduplicada y guardar sus resultados aparte, añade `--data-dir data/preprocessed/parasites-1s07h-dedup --run-name fasterrcnn_v2_dedup`.

El script muestra una barra por lote y un resumen al cerrar cada época. Los checkpoints quedan en `models/trained/fasterrcnn_v2/<run_name>/`. Los pesos originales descargados están en `models/pretrained/`. El historial y los parámetros se guardan en `results/metrics/fasterrcnn_v2/<run_name>/` como `training.csv` y `settings.json`. Para continuar hasta la época 30 a partir del último checkpoint de la ejecución predeterminada:

```bash
python scripts/training/train_faster_rcnn.py --resume models/trained/fasterrcnn_v2/baseline/last_checkpoint.pth --epochs 30
```

El learning rate usa `LinearLR` con `start_factor=0.001`, `end_factor=1.0` y `warmup_batches=1000`. El warmup va de `learning_rate * start_factor` a `learning_rate * end_factor` y dura `min(warmup_batches, lotes_por_época - 1)` lotes de la primera época. Después de cada época avanza `CosineAnnealingLR` con `T_max=epochs` hasta cero. Sigue la [referencia de detección de torchvision](https://github.com/pytorch/vision/tree/v0.23.0/references/detection). Puedes cambiar el warmup en el notebook o con `--start-factor`, `--end-factor` y `--warmup-batches`.

El checkpoint guarda el estado del scheduler. Al reanudar con el mismo número de épocas continúa la curva guardada. Si amplías las épocas o cargas un checkpoint anterior al scheduler, recalcula el learning rate para la posición actual en la nueva curva. El historial incluye el learning rate al cerrar cada época antes del siguiente paso del coseno.

`val_loss` se calcula en una pasada adicional por validación, en FP32 y con las estadísticas de BatchNorm congeladas. Las épocas anteriores a esta incorporación tendrán esa columna vacía cuando se reanude una ejecución.

Para entrenar desde Jupyter, abre [`03_faster_rcnn_training.ipynb`](../notebooks/03_faster_rcnn_training.ipynb) con el kernel **VISION P5**. Pon la ruta de tu dataset en la celda **Datos**, ajusta los parámetros y ejecuta la celda de entrenamiento.

`model` queda con los pesos de la época con mayor mAP de validación y `history` contiene las métricas por época. Para recuperar ese modelo en otra sesión, usa `load_trained_model`. Cuando termines de ajustar el entrenamiento, puedes medir el modelo final en test:

```python
from parasite_detection.faster_rcnn import create_dataloaders, evaluate_model

test_loader, = create_dataloaders(config, splits=("test",))
test_metrics = evaluate_model(model, test_loader)
test_metrics
```

## Descarga y preparación del dataset

| Archivo | Función | Uso |
|---|---|---|
| [`download_dataset.py`](data/download_dataset.py) | Descarga los Parquet originales de Hugging Face según la revisión y los checksums fijados en `configs/dataset.json`. Los guarda en `data/raw/parasites-1s07h/`. | `just dataset` |
| [`prepare_dataset.py`](data/prepare_dataset.py) | Extrae las imágenes y exporta las anotaciones a formatos COCO y YOLO en `data/converted/parasites-1s07h/`. Conserva las particiones originales. | `just prepare` |
| [`deduplicate_train.py`](data/deduplicate_train.py) | Genera `data/preprocessed/parasites-1s07h-dedup/`, quitando del train imágenes casi duplicadas de validation o test. Mantiene validation y test y guarda el registro en `results/metrics/deduplication/train_removed_near_duplicates.csv`. El umbral predeterminado es `0.90`. | `python scripts/data/deduplicate_train.py` |

Flujo inicial: `just dataset` → `just prepare`.

## Ordenación y etiquetado visual

| Archivo | Función | Uso |
|---|---|---|
| [`order_train_images_by_similarity.py`](labeling/order_train_images_by_similarity.py) | Extrae embeddings DINOv2 ViT-S/14 del train depurado, calcula un recorrido por similitud visual y lo reparte entre Alejandro, Rodrigo y Lucas. Requiere exactamente las 1.157 imágenes depuradas; usa CUDA por defecto. Guarda los embeddings en `.cache/visual-labels/` y el orden y lotes en `results/visual-labels/dinov2_train_order.json`. | `python scripts/labeling/order_train_images_by_similarity.py` |
| [`classify_train_images.py`](labeling/classify_train_images.py) | Inicia el servidor local de la herramienta de etiquetado. Lee únicamente las imágenes del train depurado y requiere que exista el manifiesto de ordenación. | `just label` |
| [`visual_labeler/`](labeling/visual_labeler/) | Contiene la página, la lógica de interacción y los estilos que sirve el clasificador. No se ejecutan directamente. | Se cargan al iniciar `classify_train_images.py`. |
| [`compare_train_augmentations.py`](labeling/compare_train_augmentations.py) | Inicia una web de solo lectura para comparar cada imagen original con sus variantes aumentadas. Muestra las etiquetas visuales clasificadas y las transformaciones aplicadas a cada variante. | `python scripts/labeling/compare_train_augmentations.py` |
| [`augmentation_viewer/`](labeling/augmentation_viewer/) | Interfaz de comparación de originales y variantes aumentadas; no se ejecuta directamente. | Se carga al iniciar `compare_train_augmentations.py`. |

Para registrar las transformaciones aplicadas en un dataset aumentado ya generado, actualiza sus metadatos COCO sin volver a escribir las imágenes:

```bash
python scripts/create_augmented_dataset.py --write-transform-metadata
python scripts/labeling/compare_train_augmentations.py
```

La web enlaza las etiquetas de `results/visual-labels/train_dedup_labels.json` por el nombre del original; las imágenes sin etiqueta aparecen como **Sin clasificar**. Si cambias etiquetas mientras está abierta, pulsa **Actualizar etiquetas**. El comparador es de solo lectura y no modifica las clasificaciones.

Para cambiar los parámetros de la ordenación, consulta `python scripts/labeling/order_train_images_by_similarity.py --help`. La primera ejecución descarga el código y los pesos preentrenados de DINOv2 mediante Torch Hub.

## Preparación y notebooks

```bash
bash scripts/environment/setup_env.sh cu128
source .venv/bin/activate
just weights
just dataset
just prepare
jupyter lab
```

Con Jupyter abierto, selecciona el kernel **VISION P5** y continúa con los notebooks de `notebooks/`.
