# Datos originales

Esta carpeta guarda los Parquet originales de `Francesco/parasites-1s07h` en `parasites-1s07h/`, junto con `source.json`, que registra su procedencia.

Desde la raíz del repositorio:

```bash
python3 scripts/download_dataset.py
```

La revisión y los checksums están fijados en `configs/dataset.json`. El descargador reutiliza los archivos completos. Las particiones de origen contienen 1484 imágenes de entrenamiento, 215 de validación y 411 de test.

Conserva estos archivos sin modificarlos. La extracción de imágenes y la conversión de anotaciones se guardan en `data/converted/`. Los Parquet y `source.json` no se suben a Git.
