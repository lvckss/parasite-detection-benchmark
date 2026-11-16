# Datos convertidos

Esta carpeta contiene `parasites-1s07h/`, generado a partir de los Parquet de `data/raw/`:

```bash
.venv/bin/python scripts/prepare_dataset.py
```

La conversión utiliza PyArrow para leer los Parquet y Pillow para consultar las dimensiones y el formato de las imágenes. Genera:

- `images/{train,val,test}/`: imágenes extraídas con nombres basados en su ID. Se guardan sus bytes originales, sin recomprimir, redimensionar ni aplicar aumentos.
- `annotations/instances_{train,val,test}.json`: anotaciones COCO con cajas `[x, y, ancho, alto]` en píxeles y clases 1–8.
- `labels/{train,val,test}/`: anotaciones YOLO con cajas `[centro_x, centro_y, ancho, alto]` normalizadas y clases 0–7.
- `dataset.yaml`: rutas y clases para YOLO.
- `class_mapping.json`: correspondencia de clases entre ambos formatos.
- `summary.json`: recuentos de imágenes y objetos por partición y especie.

Las dos representaciones describen los mismos objetos y comparten las mismas imágenes. Se conservan las particiones de los Parquet, nombrando `validation` como `val`.

Esta fase prepara los formatos de entrada. Los preprocesamientos experimentales que cambien las imágenes se desarrollarán por separado en `data/preprocessed/`.

Puedes regenerar esta carpeta con el mismo comando. Si mueves el repositorio, repítelo para actualizar las rutas de `dataset.yaml`. Los archivos generados no se suben a Git.
