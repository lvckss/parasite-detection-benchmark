# Dashboard de entrenamiento

Un solo componente para Faster R-CNN, YOLO y productores de métricas ajenos a la detección. No importa torch, torchvision ni Ultralytics. Los adaptadores traducen eventos y nunca realizan backward, selección de checkpoints o evaluación.

## Uso desde un notebook

Con el entorno actualizado mediante el script de instalación habitual del proyecto:

```python
from IPython.display import display
from parasite_detection.dashboard import TrainingDashboard

dashboard = TrainingDashboard("YOLOv10-S", config.run_name, total_epochs=config.epochs)
display(dashboard.widget)
model, history = train_model(config, dashboard=dashboard)
```

El uso es idéntico con `train_model` de Faster R-CNN. Omitir `dashboard` mantiene la salida de terminal. Los notebooks de ambos modelos ya incluyen esta conexión. Los errores del observador se notifican una vez y no detienen el entrenamiento; los errores originales del entrenador sí se propagan y se muestran en el panel del notebook. No se generan archivos de reporte de error.

El botón de sol/luna de la cabecera alterna entre modo claro y oscuro en LIVE y en el informe interactivo. Inicialmente sigue el tema de Jupyter o del sistema y recuerda la elección en el navegador cuando el almacenamiento local está disponible. El tema no modifica las métricas del experimento.

La demo [05_training_dashboard_demo.ipynb](../../../notebooks/05_training_dashboard_demo.ipynb) usa datos sintéticos y permite comprobar las transiciones sin entrenar. También se puede generar un reporte desde terminal:

```bash
python scripts/training/demo_dashboard.py --output /tmp/dashboard-demo --delay 0
```

La carpeta debe estar vacía para proteger las demos anteriores. Abre `/tmp/dashboard-demo/report.html` directamente en un navegador.

## Persistencia

El directorio de métricas de cada experimento recibe tres archivos auxiliares:

| Archivo | Contenido |
|---|---|
| `dashboard_events.jsonl` | Estados, métricas por época y cierre, sin escrituras por batch |
| `dashboard_state.json` | Último estado visual, protocolo, configuración, extras y mensajes recientes |
| `report.html` | Informe autónomo con JavaScript, CSS y datos integrados, sin CDN ni servidor |

Los CSV, la configuración y los checkpoints existentes siguen siendo la fuente de verdad. El reporte se actualiza tras cada época validada y al terminar correctamente. Si ocurre un error, se conservan los archivos de la última validación sin modificarlos y el fallo queda en el output del notebook. Si ocurre antes de iniciar la ejecución, no se crea una carpeta de reportes. El historial visual no contiene imágenes del dataset ni pesos. La captura técnica conserva solo los últimos 40 mensajes.

Durante el entrenamiento, anywidget mantiene una sola salida y recibe progreso como máximo cada 250 ms, además del último batch de cada fase. Las curvas solo reciben nuevos datos al registrar una validación. Las animaciones se ejecutan en el navegador y no hacen esperar al entrenador.

Al terminar, esa salida se sustituye mediante su `display_id` por un bundle HTML y texto sin referencia a un widget. Contiene las tarjetas finales y dos PNG pequeños con las curvas. **Hay que guardar el notebook**, con Ctrl+S o mediante el guardado automático de Jupyter, para que sus outputs se escriban en el `.ipynb`. El código no puede garantizar que el frontend haya guardado el archivo. El HTML independiente sí se guarda automáticamente desde Python.

El output conserva una representación legible aunque JavaScript no se ejecute. Con CSS permitido, los selectores cambian de gráfica sin kernel; si el visor elimina también los estilos, quedan los textos y ambas imágenes. Si el entorno ejecuta módulos JavaScript en outputs HTML confiables, el mismo frontend mejora esa representación con zoom y tooltips. **No se presupone que JupyterLab, GitHub o cualquier visor permitan este contenido activo**. El enlace al informe independiente ofrece la interacción completa de forma fiable. Los visores que no muestran outputs o eliminan todas las imágenes no pueden garantizar esa representación.

Una interrupción abrupta del proceso puede impedir consolidar el output del notebook. El último reporte por época y el CSV permiten recuperar lo que ya se hubiera guardado. Un fallo de escritura en disco tampoco puede garantizar un informe final; se comunica sin comprometer el checkpoint del entrenador.

## Recuperar una ejecución

```python
from parasite_detection.paths import RESULTS

saved = TrainingDashboard.from_run(
    RESULTS / "metrics/fasterrcnn_v2/baseline",
    model_name="Faster R-CNN v2",
)
display(saved.widget)
saved.export_html("/tmp/fasterrcnn-report.html")
```

`from_run` lee `training.csv` y `settings.json`. Reutiliza el estado visual para el estado final, tiempos, checkpoint y métricas opcionales que no están en el CSV; los valores del CSV prevalecen. No escribe ni entrena. Los CSV antiguos sin metadatos de cierre muestran «Historial guardado», porque el número de filas no prueba que hubiera una finalización correcta. Si conoces el resultado, puedes indicarlo explícitamente con `outcome="completed"` o `outcome="early_stopped"`.

## Contrato común

```python
dashboard.bind(directory)
dashboard.emit("start", total_epochs=20, evaluation={"label": "Mi evaluador"})
dashboard.emit("status", phase="training", phase_label="Entrenando", epoch=1)
dashboard.progress(batch=1, batches=100)
dashboard.emit("epoch", metrics={"epoch": 1, "map": 0.3, "map50": 0.5,
                                 "train_loss": 2.0, "val_loss": 2.3, "lr": 0.001})
dashboard.emit("finish", outcome="completed", checkpoint="ruta/al/mejor/checkpoint")
```

El productor escribe primero sus métricas en su almacenamiento habitual y después emite el evento. `TrainingEvent` permite transportar el mismo contrato como objeto. Los valores no finitos se convierten en ausentes, nunca en ceros inventados. Se admiten componentes de pérdida, learning rates y extras como claves opcionales de cada fila.

Para añadir una vista, sin duplicar la visualización:

```python
dashboard.add_view("lr", "Learning rate", [
    {"key": "lr", "label": "Learning rate", "color": "#25b4a4"},
])
```

Las magnitudes y componentes de pérdida pertenecen a cada modelo. Faster R-CNN no publica precision/recall en su evaluador actual, por lo que aparecen como ausentes. En YOLO, mAP y mAP50 proceden de COCOeval; precision/recall proceden de Ultralytics y las tarjetas lo indican. Se registran rama one-to-one, ausencia de NMS, confianza 0.001, max detections, EMA y versión. La validación final del mejor checkpoint no se incorpora como otra época. El dashboard conserva la selección de best del entrenador, incluido el desempate de YOLO.

## Implementación y límites

- `events.py` define el contrato y la persistencia JSON sin dependencias de los modelos
- `adapters.py` observa el bucle propio y los callbacks oficiales de Ultralytics 8.4.215
- `dashboard.py` reduce eventos, conecta el widget, reconstruye CSV y exporta reportes
- `frontend.js` y `style.css` comparten el diseño LIVE/REPORT y mantienen los nodos SVG

Se añade únicamente **anywidget** a las dependencias directas. Usa la infraestructura de ipywidgets ya instalada, compatible con JupyterLab 4 y Notebook moderno con soporte de widgets. No requiere instalar un servidor de la aplicación. El frontend usa SVG con tooltips, desplazamiento horizontal, zoom con Ctrl/⌘ + rueda y doble clic para restablecer. No ofrece todas las herramientas de Plotly, como selección de puntos o su barra de exportación.

Las curvas interpolan sus puntos y sus dominios durante 500 ms; los números durante 400 ms y el carrusel durante 300 ms. Cada actualización cancela la anterior. El zoom horizontal y los nodos de las gráficas se conservan al cambiar de vista o recibir nuevas épocas. Las gráficas ocultas también se actualizan. En REPORT se dibujan directamente los valores finales y se mantienen solo las transiciones de navegación. Se respeta `prefers-reduced-motion`.

Se eligió SVG para evitar incluir una biblioteca grande en cada output y para controlar conjuntamente puntos y ejes. La [documentación de Plotly](https://plotly.com/javascript/animations/) limita las transiciones suaves a scatter y describe una limitación al animar datos y layout simultáneamente. No se utiliza FigureWidget ni se atribuyen a Plotly funciones inexistentes. La [API de anywidget](https://anywidget.dev/en/getting-started/) permite compartir este módulo entre el widget y el HTML.

Guardar el estado del widget no basta para un reporte independiente: el [embedding de ipywidgets](https://ipywidgets.readthedocs.io/en/latest/embedding.html) requiere un gestor y recursos JavaScript. Además, [JupyterLab aplica reglas de confianza a los outputs](https://jupyterlab.readthedocs.io/en/stable/user/notebook.html#trusting-notebooks). Por eso el reporte no depende de restaurar widgets y conserva imágenes como alternativa.

## Comprobación

```bash
python -m unittest discover -s tests -p 'test_training_dashboard.py' -v
```

Incluye reconstrucción desde CSV, autoridad de los datos, fallos, progreso limitado, captura técnica, callbacks y una comparación exacta de pérdidas, pesos, momentum, scheduler y mejor época de un detector diminuto con y sin observador.

Comprobaciones optativas con procesos locales y con el entrenador nativo:

```bash
RUN_DASHBOARD_NOTEBOOK_CHECK=1 python -m unittest discover -s tests -p 'test_training_dashboard.py' -v
RUN_YOLO_DASHBOARD_CHECK=1 python -m unittest discover -s tests -p 'test_yolo_dashboard_integration.py' -v
```

La primera ejecuta una simulación, guarda y vuelve a leer el `.ipynb` después de detener el kernel, y comprueba que el output final no depende de un widget. La segunda requiere los pesos y el dataset depurado locales: valida las 1157 imágenes, acorta el muestreo de entrenamiento a cuatro imágenes durante dos épocas a 64 px y conserva las 215 de validación. Usa carpetas temporales para caches, métricas y checkpoints; compara exactamente el historial y todos los tensores de `best.pt` con y sin dashboard. Este control no representa un experimento del benchmark.

Para revisar la persistencia en tu instalación:

1. Ejecuta la demo hasta `Completed` y guarda el notebook
2. Detén el kernel, cierra el notebook y vuelve a abrirlo sin ejecutar sus celdas
3. Comprueba las tarjetas y las dos gráficas guardadas
4. Abre `report.html` directamente, también sin conexión, y prueba navegación, zoom y tooltips
5. Ejecuta la celda de reconstrucción desde los archivos guardados

No se necesita volver a entrenar un detector para estas comprobaciones.
