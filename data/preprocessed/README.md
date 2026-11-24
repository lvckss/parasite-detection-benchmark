# `preprocessed/` · Nuestros experimentos

Aquí irán las variantes de preprocesamiento que probemos a partir de [`converted/`](../converted/README.md).

> [!NOTE]
> Los experimentos de aumentación offline se generan en `parasites-1s07h-augmented/`.

Para generar el dataset de augmentación:

```bash
python scripts/prepare_dataset.py
python scripts/create_augmented_dataset.py
```

El primer comando es necesario si `data/converted/parasites-1s07h/` aún no
contiene las imágenes de las tres particiones. El generador lee
`configs/augmentation.json`, conserva los originales de `train`,
añade las variantes configuradas y copia `val`/`test` sin alteraciones. Las
imágenes y anotaciones originales de `data/converted/` no se modifican. El
script no entrena ningún modelo.

Cuando preparemos otra variante:

- Guardamos los parámetros usados para poder repetirla.
- Conservamos las particiones de entrenamiento, validación y test.
- Si cambia la geometría de la imagen, ajustamos también las cajas COCO y YOLO.

El aumento online no requiere duplicar el dataset. En cambio, esta variante offline
se materializa porque necesitamos inspeccionar y reutilizar las imágenes y
anotaciones aumentadas antes de entrenar.

Los datos que generemos aquí están ignorados por Git.
