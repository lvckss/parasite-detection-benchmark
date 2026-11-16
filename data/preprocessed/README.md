# `preprocessed/` · Nuestros experimentos

Aquí irán las variantes de preprocesamiento que probemos a partir de [`converted/`](../converted/README.md).

> [!NOTE]
> **Pendiente.** Por ahora solo está este README. Los datos y las transformaciones los añadiremos durante la práctica.

Cuando preparemos una variante:

- Guardamos los parámetros usados para poder repetirla.
- Conservamos las particiones de entrenamiento, validación y test.
- Si cambia la geometría de la imagen, ajustamos también las cajas.

Si la transformación se aplica al cargar las imágenes durante el entrenamiento, basta con guardar su configuración en el experimento. No hace falta duplicar el dataset en disco.

Los datos que generemos aquí están ignorados por Git.
