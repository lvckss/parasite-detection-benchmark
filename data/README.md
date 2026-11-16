# Fases de los datos

| Carpeta | Propósito |
|---|---|
| [raw/](raw/README.md) | Parquet originales descargados de Hugging Face, sin modificar. |
| [converted/](converted/README.md) | Imágenes extraídas y anotaciones COCO/YOLO para los mismos objetos. |
| [preprocessed/](preprocessed/README.md) | Reservada para las transformaciones que implemente el equipo más adelante. |

El flujo disponible actualmente es `raw → converted`. La conversión conserva las imágenes originales y las particiones publicadas como Parquet. Los modelos pueden utilizar directamente `converted/`.

Los datos de estas carpetas son locales y están ignorados por Git; sus README se comparten con el equipo.
