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
│   └── visual_labeler/
├── training/
│   ├── train_faster_rcnn.py
│   └── train_yolo.py
└── evaluation/
    └── evaluate_faster_rcnn.py
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

## Evaluación de Faster R-CNN

[`evaluation/evaluate_faster_rcnn.py`](evaluation/evaluate_faster_rcnn.py) carga `best_model.pth` de una ejecución, predice imagen a imagen sobre una partición y la evalúa con `pycocotools`. En `results/metrics/fasterrcnn_v2/<run_name>/` guarda `<split>_metrics.json` (las 12 métricas COCO, umbrales y latencia con lote 1), `<split>_per_class.csv` (AP y AP50 por especie), `<split>_pr_curves.csv` (curvas PR por clase a IoU 0,5 y 0,75) y `<split>_predictions.json` (detecciones en formato COCO para análisis de errores, ajuste de umbrales o ensembles sin repetir la inferencia).

```bash
python scripts/evaluation/evaluate_faster_rcnn.py --split test
python scripts/evaluation/evaluate_faster_rcnn.py --split val --run-name baseline --device cpu
```

A diferencia de `evaluate_model`, guarda las predicciones y el tiempo de inferencia. [`04_faster_rcnn_test_analysis.ipynb`](../notebooks/04_faster_rcnn_test_analysis.ipynb) las usa para las figuras de `results/figures/fasterrcnn_v2/<run_name>/` con las funciones de `src/parasite_detection/analysis.py`.

Por defecto usa el dataset deduplicado, cuyo val y test coinciden con los originales. La latencia depende del hardware y se registra en el JSON: compara tiempos de modelos solo si se midieron en la misma máquina.

## Entrenamiento de YOLOv10-S

[`training/train_yolo.py`](training/train_yolo.py) usa las funciones de [`src/parasite_detection/yolo.py`](../src/parasite_detection/yolo.py). Comparte con RCNN los dataloaders COCO, el paso de entrenamiento, la medición de pérdida de validación y la evaluación con `pycocotools`. Por defecto exige las mismas 1.157 imágenes definitivas del train depurado, comprobando sus identificadores contra el manifiesto de ordenación visual. No mezcla train, validation ni test.

```bash
python scripts/training/train_yolo.py --device cuda
```

Los valores iniciales coinciden con el experimento registrado de RCNN: 20 épocas, batch 2, SGD con LR `0.0025`, momentum `0.9`, weight decay `0.0005`, imagen de 640 px, semilla 42 y paciencia 10. El warmup usa los mismos factores `0.001` y `1.0` y se limita a 1.000 lotes de la primera época; después se aplica `CosineAnnealingLR`. `--patience 0` desactiva la parada temprana y `--no-amp` desactiva la precisión mixta.

El adaptador carga los pesos de `models/pretrained/yolov10s.pt`, transfiere los tensores compatibles y adapta la clasificación a ocho clases. Conserva las dos ramas de asignación y la pérdida `E2EDetectLoss` de YOLOv10, con ganancias `box=7.5`, `cls=0.5` y `dfl=1.5`. Se pueden cambiar con `--box-gain`, `--cls-gain` y `--dfl-gain`. La entrada se escala conservando su proporción y se completa con relleno hasta un cuadrado. No se aplican aumentos aleatorios, EMA ni acumulación de gradientes.

La evaluación convierte las cajas a las coordenadas originales y las clases YOLO 0–7 a COCO 1–8. Usa un umbral de confianza de `0.05` y hasta 100 detecciones por imagen, como RCNN; la rama de inferencia de YOLOv10 no necesita NMS. El mejor modelo y la parada temprana se deciden por mAP COCO de validación. Las pérdidas son propias de cada arquitectura y sus valores absolutos no son comparables entre detectores.

Los checkpoints se guardan en `models/trained/yolov10s/<run_name>/` como `best_model.pth` y `last_checkpoint.pth`. Son checkpoints de este módulo y se cargan con `load_trained_model`, no con `ultralytics.YOLO`. Las métricas quedan en `results/metrics/yolov10s/<run_name>/training.csv`, con las mismas columnas `epoch`, `train_loss`, `val_loss`, `map`, `map50` y `lr`. `settings.json` registra los parámetros y las versiones de PyTorch y Ultralytics. Para un nuevo experimento, cambia `--run-name`.

```bash
python scripts/training/train_yolo.py \
    --resume models/trained/yolov10s/baseline/last_checkpoint.pth --epochs 30
```

La reanudación recupera modelo, optimizador, scheduler, scaler e historial. Si amplías las épocas, recalcula la curva coseno igual que RCNN. Conserva los parámetros del experimento y las versiones de las bibliotecas.

La inferencia activa expresamente la rama `one2one` y comprueba que devuelve detecciones con seis valores por fila. Los checkpoints anteriores a esta corrección pueden cargarse para reevaluar sus pesos, pero sus mAP registrados no son válidos. Para generar un historial comparable, repite el entrenamiento con otro `run_name`; la reanudación rechaza esos historiales antiguos.

Desde un notebook puedes usar la misma interfaz de configuración y entrenamiento:

```python
from parasite_detection.yolo import TrainConfig, train_model

config = TrainConfig(device="cuda", run_name="baseline")
model, history = train_model(config)
```

Test se evalúa expresamente al terminar los ajustes, sin intervenir en la selección del modelo:

```python
from parasite_detection.yolo import create_dataloaders, evaluate_model, load_trained_model

model = load_trained_model(device="cuda")
test_loader, = create_dataloaders(config, splits=("test",))
test_metrics = evaluate_model(model, test_loader)
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
