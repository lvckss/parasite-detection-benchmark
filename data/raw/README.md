# `raw/` · Originales

Aquí se descargan los tres Parquet de [Francesco/parasites-1s07h](https://huggingface.co/datasets/Francesco/parasites-1s07h). Cada uno guarda las imágenes y sus anotaciones.

## Descargar

Desde la raíz del repo:

```bash
python3 scripts/download_dataset.py
```

```text
raw/
├── README.md
└── parasites-1s07h/
    ├── train-*.parquet
    ├── validation-*.parquet
    ├── test-*.parquet
    └── source.json
```

La revisión y los checksums están fijados en [`configs/dataset.json`](../../configs/dataset.json). `source.json` guarda esa información junto a la descarga. Al repetir el comando, se reutilizan los archivos completos.

> [!IMPORTANT]
> Conserva estos archivos tal como se descargan. La extracción y la conversión se escriben en [`converted/`](../converted/README.md); los cambios experimentales irán en [`preprocessed/`](../preprocessed/README.md).

Los Parquet y `source.json` son locales y no se suben a Git.
