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
├── training/
│   ├── train_faster_rcnn.py
│   └── train_yolo.py
└── evaluation/
    ├── evaluate_faster_rcnn.py
    ├── evaluate_yolo.py
    └── benchmark_latency.py
```

## Entorno y pesos de los modelos

| Archivo | Función | Uso |
|---|---|---|
| [`setup_env.sh`](environment/setup_env.sh) | Crea o actualiza `.venv`, instala las dependencias y registra el kernel de Jupyter. Acepta `cpu` o `cu128`. | `bash scripts/environment/setup_env.sh cu128` |
| [`download_weights.py`](models/download_weights.py) | Descarga a `models/pretrained/` los pesos COCO de Faster R-CNN v2 y YOLOv10-S. Omite los archivos existentes y mueve los de la estructura anterior a esta carpeta; `--force` los reemplaza y `--only fasterrcnn` o `--only yolo` descarga un solo modelo. | `just weights` |

Los pesos preentrenados y archivos descargados se mantienen localmente. Los checkpoints entrenados de los baselines se comparten mediante Git LFS; usa `git lfs pull` para descargarlos.

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

[`training/train_yolo.py`](training/train_yolo.py) lanza el entrenador oficial de Ultralytics desde [`src/parasite_detection/yolo.py`](../src/parasite_detection/yolo.py). Usa la versión `8.4.215` fijada en `uv.lock`. El motor oficial gestiona la transferencia COCO, la pérdida, los grupos del optimizador, Nesterov, acumulación de gradientes, clipping y EMA.

```bash
python scripts/training/train_yolo.py --device cuda
```

El nombre predeterminado es `baseline_ultralytics`, separado del baseline anterior con el bucle personalizado. Antes de entrenar se comprueba que las imágenes, anotaciones COCO y etiquetas YOLO coincidan en train y validation, incluidas las cajas y clases. Test se valida al evaluarlo. El train debe ser el del manifiesto depurado o una variante creada por `create_augmented_dataset.py`, con sus imágenes fuente y `augmentation_config.json` coherentes. El dataloader comprueba además que Ultralytics conserve todos los archivos anotados. Puedes seleccionar el mismo dataset para ambos entrenadores con `PARASITE_DATA_DIR` o indicar `--data-dir`.

El baseline no aplica aumentos aleatorios. El dataloader de train usa exclusivamente `LetterBox` y `Format`, y se comprueba esa lista de transformaciones. Los parámetros de augmentations se guardan a cero en los argumentos de la ejecución.

| Parámetro | Valor predeterminado |
|---|---|
| Épocas y batch | 100 y 8 |
| Entrada y semilla | 640 px y 42 |
| Optimizador | SGD, LR 0.01, momentum 0.9, weight decay 0.0005 |
| Batch nominal para acumulación | 64 |
| Warmup | 3 épocas, momentum inicial 0.8, LR inicial de biases 0.1 |
| Scheduler | Coseno, fracción final de LR 0.01 |
| Paciencia | 20 épocas sin mejorar el mAP COCO |
| Precisión | AMP en GPU |

La pérdida se elige mediante la implementación oficial de la arquitectura; en esta versión es `E2ELoss`. No se divide su escalar de backward por el batch. `--patience 0` desactiva la parada temprana y `--no-amp` desactiva AMP. Consulta el resto de parámetros con `python scripts/training/train_yolo.py --help`.

La validación fija un lienzo cuadrado del tamaño indicado, tanto durante el entrenamiento como al evaluar por separado. Usa la EMA y la cabeza one-to-one con `nms=False`, confianza `0.001` y hasta 100 detecciones por imagen. `CocoValidator` conserva el postprocesado oficial, convierte las predicciones a los IDs y coordenadas originales y aplica `pycocotools.COCOeval`. Su mAP determina tanto el mejor checkpoint como la parada temprana. La consola muestra además las métricas nativas de Ultralytics; el resumen identificado como **COCO** corresponde al evaluador común con RCNN.

Los checkpoints oficiales quedan en `models/trained/yolov10s/<run_name>/weights/` como `best.pt` y `last.pt`. Los de `baseline` se comparten mediante Git LFS. El directorio también contiene `args.yaml`, `dataset.yaml` con las rutas locales y el `results.csv` nativo; estos archivos auxiliares y las demás ejecuciones permanecen locales.

El historial para los notebooks queda en `results/metrics/yolov10s/<run_name>/training.csv`. Conserva `epoch`, `train_loss`, `val_loss`, `map`, `map50` y `lr`; añade las componentes de pérdida, los LR por grupo y el mAP nativo bajo `ultralytics_map` y `ultralytics_map50`. `map` y `map50` siempre son COCO. Las pérdidas suman las componentes registradas por el motor oficial y no representan su escalar de backward; no deben compararse numéricamente con las del bucle antiguo. `settings.json` guarda los argumentos efectivos, versiones y criterios de evaluación y selección. `best_validation.json` contiene la evaluación final de los pesos guardados; puede diferir ligeramente del historial porque la serialización nativa usa FP16.

Para reanudar una ejecución interrumpida, conserva el dataset, el nombre, el horizonte de épocas y los hiperparámetros originales:

```bash
python scripts/training/train_yolo.py \
    --resume models/trained/yolov10s/baseline_ultralytics/weights/last.pt
```

Ultralytics recupera optimizador, scaler y EMA. Al finalizar normalmente, el motor retira el optimizador de los checkpoints: esos archivos sirven para inferencia, pero no para reanudar esta misma ejecución. Para otro experimento, usa un nuevo `run_name`. Los antiguos `.pth` se pueden leer con `load_trained_model` para evaluar sus pesos, pero no reanudar con el nuevo motor.

Desde [`04_yolov10s_training.ipynb`](../notebooks/04_yolov10s_training.ipynb), reinicia el kernel para cargar el módulo actualizado:

```python
from parasite_detection.yolo import TrainConfig, train_model

config = TrainConfig(device="cuda", run_name="baseline_ultralytics")
model, history = train_model(config)
```

`model` es una instancia de `ultralytics.YOLO` con los mejores pesos seleccionados por COCO. `history` sigue disponible para las gráficas. Los parámetros antiguos `start_factor`, `end_factor` y `warmup_batches` se sustituyen por `warmup_epochs`, `warmup_momentum` y `warmup_bias_lr`.

Test se evalúa expresamente cuando hayan terminado los ajustes:

```python
from parasite_detection.yolo import evaluate_model, load_trained_model

model = load_trained_model(device="cuda")
test_metrics = evaluate_model(model, data_dir=config.data_dir, split="test", device="cuda")
```

## Evaluación de YOLOv10-S

[`evaluation/evaluate_yolo.py`](evaluation/evaluate_yolo.py) evalúa `best.pt` igual que `evaluate_faster_rcnn.py` y guarda los mismos cuatro archivos con el mismo formato en `results/metrics/yolov10s/<run_name>/`: `<split>_metrics.json`, `<split>_per_class.csv`, `<split>_pr_curves.csv` y `<split>_predictions.json` (IDs COCO 1–8). Las funciones comunes de COCOeval están en [`src/parasite_detection/evaluation.py`](../src/parasite_detection/evaluation.py).

```bash
python scripts/evaluation/evaluate_yolo.py --split val
python scripts/evaluation/evaluate_yolo.py --split test
```

Las predicciones salen de `CocoValidator`, el mismo letterbox cuadrado y la misma salida one-to-one sin NMS (confianza `0.001`, 100 detecciones) con los que se eligió el checkpoint, así que el mAP de val debe coincidir con `best_validation.json`. La latencia se mide en una pasada aparte con lote 1: preprocesado, red y postprocesado, sin la lectura de la imagen y sin las 3 primeras imágenes. Usa los mismos hilos de CPU que torch por defecto (Ultralytics los limita a 8), para compararla con Faster R-CNN medido en la misma máquina. [`05_yolov10s_test_analysis.ipynb`](../notebooks/05_yolov10s_test_analysis.ipynb) genera las figuras en `results/figures/yolov10s/<run_name>/`.

## Comparación de latencias

La latencia de las pasadas de evaluación varía mucho en un portátil (energía, temperatura). Para comparar modelos, [`evaluation/benchmark_latency.py`](evaluation/benchmark_latency.py) carga varios detectores y los alterna imagen a imagen, cambiando el orden en cada imagen, sobre las mismas imágenes de test (50 por defecto, semilla 42, más 3 de calentamiento). Todos usan los mismos hilos de CPU, lote 1 y la imagen ya en memoria. Guarda la mediana, p25, p75, p95 y media en `results/metrics/latency/<split>_<device>.json` y el tiempo de cada imagen en un CSV con el mismo nombre.

```bash
python scripts/evaluation/benchmark_latency.py
python scripts/evaluation/benchmark_latency.py --model fasterrcnn_v2:baseline --model yolov10s:baseline --images 100
```

Entrena cada experimento donde quieras, pero mide todas las latencias en la misma máquina. Las mejoras que no cambian la arquitectura ni la inferencia (aumentos, muestreo, pérdidas) tienen la misma latencia que su baseline.

## Dashboard de entrenamiento

[`demo_dashboard.py`](training/demo_dashboard.py) genera un informe autocontenido con métricas simuladas, sin cargar modelos ni datos:

```bash
python scripts/training/demo_dashboard.py --output /tmp/dashboard-demo --delay 0
```

Abre `/tmp/dashboard-demo/report.html` directamente en un navegador. La carpeta debe estar vacía. Para probar el modo LIVE, usa [`05_training_dashboard_demo.ipynb`](../notebooks/05_training_dashboard_demo.ipynb). Ambos entrenadores admiten `train_model(config, dashboard=dashboard)`; el historial CSV y los checkpoints siguen siendo los originales. [API, persistencia y pruebas](../src/parasite_detection/dashboard/README.md).

## Descarga y preparación del dataset

| Archivo | Función | Uso |
|---|---|---|
| [`download_dataset.py`](data/download_dataset.py) | Descarga los Parquet originales de Hugging Face según la revisión y los checksums fijados en `configs/dataset.json`. Los guarda en `data/raw/parasites-1s07h/`. | `just dataset` |
| [`prepare_dataset.py`](data/prepare_dataset.py) | Extrae las imágenes y exporta las anotaciones a formatos COCO y YOLO en `data/converted/parasites-1s07h/`. Conserva las particiones originales. | `just prepare` |
| [`deduplicate_train.py`](data/deduplicate_train.py) | Genera `data/preprocessed/parasites-1s07h-dedup/`, quitando del train imágenes casi duplicadas de validation o test. Mantiene validation y test y guarda el registro en `results/metrics/deduplication/train_removed_near_duplicates.csv`. El umbral predeterminado es `0.90`. | `python scripts/data/deduplicate_train.py` |
| [`create_augmented_dataset.py`](create_augmented_dataset.py) | Genera `data/preprocessed/parasites-1s07h-augmented-v1/` desde el train deduplicado, conserva los originales, añade las variantes de `configs/augmentation.json` y deja val/test intactos. | `python scripts/create_augmented_dataset.py` |

Flujo inicial: `just dataset` → `just prepare`.

## Ordenación y etiquetado visual

| Archivo | Función | Uso |
|---|---|---|
| [`order_train_images_by_similarity.py`](labeling/order_train_images_by_similarity.py) | Extrae embeddings DINOv2 ViT-S/14 del train depurado, calcula un recorrido por similitud visual y lo reparte entre Alejandro, Rodrigo y Lucas. Requiere exactamente las 1.157 imágenes depuradas; usa CUDA por defecto. Guarda los embeddings en `.cache/visual-labels/` y el orden y lotes en `results/visual-labels/dinov2_train_order.json`. | `python scripts/labeling/order_train_images_by_similarity.py` |
| [`classify_train_images.py`](labeling/classify_train_images.py) | Inicia el servidor local de la herramienta de etiquetado. Lee únicamente las imágenes del train depurado y requiere que exista el manifiesto de ordenación. | `just label` |
| [`compare_train_augmentations.py`](labeling/compare_train_augmentations.py) | Abre la web local para revisar originales y variantes de una carpeta aumentada que incluya `applied_transforms`. | `python scripts/labeling/compare_train_augmentations.py --data-dir data/preprocessed/parasites-1s07h-augmented-v1` |
| [`visual_labeler/`](labeling/visual_labeler/) | Contiene la página, la lógica de interacción y los estilos que sirve el clasificador. No se ejecutan directamente. | Se cargan al iniciar `classify_train_images.py`. |
| [`augmentation_viewer/`](labeling/augmentation_viewer/) | Contiene la página, la lógica de comparación y los estilos del visor de augmentaciones. No se ejecutan directamente. | Se sirven al iniciar `compare_train_augmentations.py`. |

Para cambiar los parámetros de la ordenación, consulta `python scripts/labeling/order_train_images_by_similarity.py --help`. La primera ejecución descarga el código y los pesos preentrenados de DINOv2 mediante Torch Hub. Ambos servidores se detienen con `Ctrl+C`; puedes consultar sus opciones con `--help`.

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
