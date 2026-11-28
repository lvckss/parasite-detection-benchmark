# `preprocessed/` · Nuestros experimentos

Aquí irán las variantes de preprocesamiento que probemos a partir de [`converted/`](../converted/README.md).

> [!NOTE]
> Los experimentos de aumentación offline se generan en `parasites-1s07h-augmented/`.

Genera primero la variante deduplicada para no reintroducir en train casi duplicados de val/test. Después crea la aumentación desde esa variante:

```bash
python scripts/data/prepare_dataset.py
python scripts/data/deduplicate_train.py
python scripts/create_augmented_dataset.py
```

La preparación solo hace falta si `data/converted/parasites-1s07h/` aún no
contiene las imágenes de las tres particiones. El generador usa por defecto
`data/preprocessed/parasites-1s07h-dedup/` como origen y crea
`data/preprocessed/parasites-1s07h-augmented-v1/`. Lee
`configs/augmentation.json`, conserva los originales de `train`, añade las
variantes configuradas y copia `val`/`test` sin alteraciones. Para otro
experimento, guarda la configuración y el resultado en rutas nuevas; no
sobrescribe datasets existentes. El script no entrena ningún modelo.

La carpeta `parasites-1s07h-augmented/` creada con versiones anteriores del
generador puede contener aumentaciones de las 1.484 imágenes originales de
train. No la uses para los experimentos sin duplicados: vuelve a generarla
desde `parasites-1s07h-dedup/` en una carpeta nueva.
## `parasites-1s07h-dedup/` · Train sin casi duplicados

En [`01_dataset_overview.ipynb`](../../notebooks/01_dataset_overview.ipynb) vimos que ≈ 35–38 % de val y test son casi duplicados de imágenes de train. Esta variante quita de **train** esas imágenes para que val y test midan imágenes que el modelo no ha visto. **Val y test no cambian.**

```bash
python scripts/data/deduplicate_train.py
```

- Usa la misma huella que el notebook: miniatura 32×32 en gris normalizada y correlación ≥ 0,90 (`--threshold` para cambiarlo). Por encima de 0,99 son copias exactas; entre 0,90 y 0,98 revisamos las parejas a mano y muchas son la misma foto con otro color, recorte o marca de agua. Algunas son fotos distintas que comparten el círculo del microscopio, pero perder unas pocas imágenes de train cuesta menos que dejar duplicados.
- Si cambias el umbral, cambia también `THRESHOLD` en el notebook para que el test sin duplicados use el mismo criterio.
- Compara cada imagen de train con **todas** las de val y test. Una imagen de evaluación puede tener varias copias en train, así que se quitan más imágenes de train que las 233 de evaluación afectadas.
- Con 0,90 se quitan 327 imágenes de train: quedan 1157 imágenes y 3015 objetos (antes 1484 y 3667).
- La lista está en [`results/metrics/deduplication/train_removed_near_duplicates.csv`](../../results/metrics/deduplication/train_removed_near_duplicates.csv) y los parámetros en `deduplication.json`.

La carpeta tiene la misma estructura que [`converted/`](../converted/README.md), así que basta con cambiar la ruta: `DEDUP_DATA` en `parasite_detection.paths` o su `dataset.yaml` para YOLO.

> [!IMPORTANT]
> Esta variante cambia la partición de train original. Los resultados con ella son un experimento aparte, no sustituyen a los obtenidos con los splits originales.

Cuando preparemos otra variante:

- Guardamos los parámetros usados para poder repetirla.
- Conservamos las particiones de entrenamiento, validación y test.
- Si cambia la geometría de la imagen, ajustamos también las cajas COCO y YOLO.

El aumento online no requiere duplicar el dataset. En cambio, esta variante offline
se materializa porque necesitamos inspeccionar y reutilizar las imágenes y
anotaciones aumentadas antes de entrenar.

Los datos que generemos aquí están ignorados por Git.
