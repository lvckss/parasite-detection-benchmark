# Scripts

Herramientas del proyecto, agrupadas por tarea. Ejecuta los comandos desde la raíz del repositorio, con el entorno virtual activo cuando corresponda.

## Entorno y pesos de los modelos

| Archivo | Función | Uso |
|---|---|---|
| [`setup_env.sh`](setup_env.sh) | Crea o actualiza `.venv`, instala las dependencias y registra el kernel de Jupyter. Acepta `cpu` o `cu128`. | `bash scripts/setup_env.sh cu128` |
| [`download_weights.py`](download_weights.py) | Descarga a `models/` los pesos COCO de Faster R-CNN v2 y YOLOv10-S. Omite los archivos existentes; `--force` los reemplaza y `--only fasterrcnn` o `--only yolo` descarga un solo modelo. | `just weights` |

Los pesos y archivos descargados se mantienen localmente y no se incluyen en Git.

## Descarga y preparación del dataset

| Archivo | Función | Uso |
|---|---|---|
| [`download_dataset.py`](download_dataset.py) | Descarga los Parquet originales de Hugging Face según la revisión y los checksums fijados en `configs/dataset.json`. Los guarda en `data/raw/parasites-1s07h/`. | `python scripts/download_dataset.py` |
| [`prepare_dataset.py`](prepare_dataset.py) | Extrae las imágenes y exporta las anotaciones a formatos COCO y YOLO en `data/converted/parasites-1s07h/`. Conserva las particiones originales. | `python scripts/prepare_dataset.py` |
| [`deduplicate_train.py`](deduplicate_train.py) | Genera `data/preprocessed/parasites-1s07h-dedup/`, quitando del train imágenes casi duplicadas de validation o test. Mantiene validation y test y guarda el registro en `results/metrics/train_removed_near_duplicates.csv`. El umbral predeterminado es `0.90`. | `python scripts/deduplicate_train.py` |

Flujo inicial: `download_dataset.py` → `prepare_dataset.py` →, si se necesita la variante depurada, `deduplicate_train.py`.

## Ordenación y etiquetado visual

| Archivo | Función | Uso |
|---|---|---|
| [`order_train_images_by_similarity.py`](order_train_images_by_similarity.py) | Extrae embeddings DINOv2 ViT-S/14 del train depurado, calcula un recorrido por similitud visual y lo reparte entre Alejandro, Rodrigo y Lucas. Requiere exactamente las 1.157 imágenes depuradas; usa CUDA por defecto. Guarda los embeddings en `.cache/visual-labels/` y el orden y lotes en `results/visual-labels/dinov2_train_order.json`. | `python scripts/order_train_images_by_similarity.py` |
| [`classify_train_images.py`](classify_train_images.py) | Inicia el servidor local de la herramienta de etiquetado. Lee únicamente las imágenes del train depurado y requiere que exista el manifiesto de ordenación. | `just label` |
| [`visual_labeler/`](visual_labeler/) | Contiene la página, la lógica de interacción y los estilos que sirve el clasificador. No se ejecutan directamente. | Se cargan al iniciar `classify_train_images.py`. |

Para cambiar los parámetros de la ordenación, consulta `python scripts/order_train_images_by_similarity.py --help`. La primera ejecución descarga el código y los pesos preentrenados de DINOv2 mediante Torch Hub.

## Ejemplo de preparación completa

```bash
bash scripts/setup_env.sh cu128
source .venv/bin/activate
python scripts/download_dataset.py
python scripts/prepare_dataset.py
python scripts/deduplicate_train.py
python scripts/order_train_images_by_similarity.py
just label
```
