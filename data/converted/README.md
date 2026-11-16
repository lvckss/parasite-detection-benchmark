# `converted/` · Listos para cargar

Extraemos las imágenes de los Parquet y escribimos las anotaciones que necesita cada detector. **Ambos modelos usan las mismas imágenes y los mismos objetos.**

## Convertir

Desde la raíz del repo, después de descargar los datos:

```bash
.venv/bin/python scripts/prepare_dataset.py
```

El script lee los Parquet con **PyArrow** y consulta el formato y las dimensiones de las imágenes con **Pillow**. Conserva las particiones y llama `val` a `validation`.

> [!NOTE]
> **Las imágenes mantienen sus bytes originales.** Aquí no se redimensiona, recomprime ni aplica data augmentation. Para YOLO se normalizan las coordenadas de las cajas; la imagen y el objeto señalado siguen siendo los mismos.

## Qué se genera

Todo queda dentro de `parasites-1s07h/`:

| Ruta | Contenido |
|---|---|
| `images/{train,val,test}/` | Imágenes con nombres basados en su ID |
| `annotations/instances_{train,val,test}.json` | Anotaciones COCO |
| `labels/{train,val,test}/` | Un TXT de etiquetas YOLO por imagen |
| `dataset.yaml` | Rutas y nombres de clase para YOLO |
| `class_mapping.json` | Correspondencia de etiquetas entre modelos |
| `summary.json` | Recuentos de imágenes y objetos por partición y especie |

## Cajas y clases

| | COCO / Faster R-CNN | YOLOv10-S |
|---|---|---|
| Caja guardada | `[x, y, ancho, alto]` | `[centro_x, centro_y, ancho, alto]` |
| Coordenadas | Píxeles | Normalizadas entre 0 y 1 |
| Especies | IDs **1–8** | IDs **0–7** |
| Entrada de datos | Imágenes + JSON COCO | `dataset.yaml` |

En torchvision, **0 es background**: la cabeza del detector tendrá 9 salidas para las 8 especies y el fondo. El dataset de entrenamiento deberá convertir las cajas COCO a `[x1, y1, x2, y2]` al cargarlas.

La categoría general `parasites` con ID 0 de los metadatos originales no cuenta como una novena especie. La correspondencia completa está en `class_mapping.json`.

> [!TIP]
> Si mueves el repo, vuelve a ejecutar la conversión: `dataset.yaml` contiene la ruta local de las imágenes. Puedes regenerar esta carpeta con el mismo comando.

Los archivos generados están ignorados por Git. Los futuros preprocesamientos se guardarán en [`preprocessed/`](../preprocessed/README.md).
