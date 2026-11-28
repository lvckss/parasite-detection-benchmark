# Reentrenamiento del baseline de Faster R-CNN v2

Repetición del baseline con la misma configuración (`settings.json`, semilla 42) en otra máquina con GPU. Se ejecutó con `run_name="baseline"` y se publicó en `3a5587d` sobre los archivos del baseline; aquí se conserva aparte.

- Mejor época 17, mAP@[.5:.95] en validación 0,657 (baseline: época 16, 0,663).
- Sus pesos no están en el repositorio, así que no se ha evaluado en test.
- Sirve como medida del ruido entre ejecuciones idénticas en GPU: unas 0,006 de mAP en validación. Diferencias menores entre experimentos no son concluyentes.
