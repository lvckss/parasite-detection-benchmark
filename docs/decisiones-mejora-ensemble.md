# Mejora: ensemble de Faster R-CNN y YOLOv10-S según el tipo de imagen

Responsable: Alejandro. Este documento recoge **qué se decidió, por qué y con qué datos**, para poder justificarlo en la presentación. Las cifras de validación (val) sirven para decidir; las de test solo se miden al final.

## Resumen en una frase

Combinamos las detecciones de los dos baselines con *Weighted Boxes Fusion* (WBF) y comprobamos si conviene dar a YOLO un peso distinto según si la imagen es de microscopía o no, usando un clasificador entrenado con **nuestras etiquetas manuales** del train.

**Resultado:** el ensemble mejora a Faster R-CNN en test (mAP 0,623 → 0,635, IC 95 % [+0,000, +0,026]) a cambio de un 8 % más de latencia, sobre todo porque encuentra más huevos y ajusta mejor las cajas. Ponderar por tipo de imagen no aportó nada.

## 1. Punto de partida: qué dicen los baselines

| | Faster R-CNN v2 | YOLOv10-S |
|---|---|---|
| mAP@[.5:.95] val / test | 0,663 / 0,623 | 0,612 / 0,537 |
| Latencia (CPU, mediana) | 3241 ms | 265 ms |
| Error que más mAP@0.5 resta (test) | Clasificación (+0,088) | Clasificación (+0,119) |
| Segundo error | Fondo (+0,049) | Fondo (+0,051) |

- En los dos modelos **lo que más falla es la especie**, no la localización (corregir la localización solo daría +0,004 y +0,009).
- La mayoría de errores de clasificación son **cajas secundarias de poca confianza** con otra especie sobre un huevo ya detectado (en val, de 388 errores de clasificación de Faster solo unos 12 tienen confianza ≥ 0,9).
- YOLOv10 no usa NMS y deja **384 cajas duplicadas** en test.

## 2. Opciones consideradas y por qué no se eligieron

| Opción | Por qué no |
|---|---|
| **C. Hard example mining** (idea inicial) | Faster casi memoriza el train (pérdida de train 0,037 frente a 0,16 en val), así que medir la dificultad con el modelo final daría casi cero y reforzaría sobre todo errores de anotación. Además solo es fácil de aplicar a Faster (YOLO usa el cargador de Ultralytics) y necesita 3–5 entrenamientos en Colab. Las clases ya están equilibradas (289–542 objetos), así que el muestreo por clase no aporta. |
| **C adaptada** (dificultad medida con un modelo de 4 épocas) | Resuelve la memorización, pero sigue siendo solo para Faster y depende de la GPU de Colab antes de la entrega. Preferimos una mejora que sirva para **los dos modelos**. |
| **D. TTA con rotaciones y volteos + ensemble** | El TTA usa las mismas rotaciones y volteos que la aumentación de Rodrigo (opción A): la justificación sería la misma y su efecto se confundiría con el de A. Nos quedamos solo con el ensemble. |
| **D'. Ensemble según el tipo de imagen** | **Elegida**: sirve para los dos modelos, no reentrena, ataca el error de clasificación y usa nuestro etiquetado manual dentro del sistema. |

## 3. Comprobación previa en validación (antes de implementar)

Antes de invertir tiempo comprobamos dos cosas con las predicciones ya guardadas de val.

**¿Rinden distinto los modelos según el tipo de imagen?** (tipo propuesto por el clasificador, sin revisar)

| Grupo | Imágenes | Faster | YOLO | Ventaja de Faster (IC 95 %) |
|---|---|---|---|---|
| Todas | 215 | 0,663 | 0,612 | +0,051 [+0,020, +0,090] |
| Microscopía | 166 | 0,700 | 0,655 | +0,045 [+0,004, +0,079] |
| No microscopía | 49 | 0,611 | 0,482 | +0,129 [+0,028, +0,193] |

- Faster gana en todos los tipos, así que **elegir un modelo según el tipo no tiene sentido**.
- Pero YOLO está cerca de Faster en microscopía y se hunde fuera de ella. La ventaja de Faster es 0,084 mayor fuera de microscopía (IC 95 % [−0,021, +0,156], probabilidad de no haber diferencia 0,07): **es un indicio, no algo demostrado**, porque solo hay 49 imágenes que no son de microscopía. Por eso la ponderación por tipo es una hipótesis que se contrasta, no algo que se da por hecho.

**¿Se complementan los modelos?** Huevos detectados con el umbral de F1 de cada modelo (Faster 0,90, YOLO 0,30):

| Grupo | Huevos | Ambos | Solo Faster | Solo YOLO | Ninguno |
|---|---|---|---|---|---|
| Todas | 460 | 277 | 50 | 45 | 88 |
| Microscopía | 311 | 199 | 29 | 33 | 50 |
| No microscopía | 149 | 78 | 21 | 12 | 38 |

- YOLO encuentra **45 huevos (10 %) que Faster no detecta**. Uniendo los dos se encontraría el 81 % de los huevos frente al 71 % de Faster solo. Es el argumento principal para el ensemble.
- En microscopía YOLO aporta incluso más que Faster (33 frente a 29); fuera, menos (12 frente a 21). Coherente con dar más peso a YOLO en microscopía.

## 4. Decisiones de diseño

1. **WBF en lugar de NMS para fusionar.** El NMS se queda con una caja y tira las demás; WBF promedia las cajas que se solapan ponderando por su confianza, y **rebaja la confianza de las cajas que propone un solo modelo**. Así: las cajas que ambos modelos confirman suben, los falsos positivos de un solo modelo bajan (error de fondo) y los duplicados de YOLO se unen. Referencia: Solovyev et al., *Weighted boxes fusion: Ensembling boxes from different object detection models*, Image and Vision Computing, 2021.
2. **Implementación propia de WBF** (`src/parasite_detection/ensemble.py`), validada contra la librería de los autores (`ensemble-boxes` 1.0.9): resultados idénticos en 212 de 215 imágenes de val; en las otras 3 solo cambia el orden de empates en la cuarta cifra decimal (ellos usan precisión simple). Así no añadimos dependencias y entendemos cada paso.
3. **Fusión por especie** (variante estándar de WBF): solo se fusionan cajas de la misma clase. La especie no se «vota» entre clases; lo que corrige la clasificación es que la especie que confirman los dos modelos acaba con más confianza que la que propone uno solo.
4. **Faster es el modelo de referencia (peso 1) y el peso de YOLO se ajusta** entre 0,25 y 1, porque Faster es mejor en todas las especies.
5. **Solo dos grupos: microscopía frente al resto.** Val tiene solo 49 imágenes que no son de microscopía; con cuatro tipos habría que ajustar pesos con 7 u 11 imágenes. Además, el clasificador distingue bien la microscopía (acierta 707 de 731 en train) pero confunde a menudo diapositivas con microscopía (115 de 275), que suelen ser fotos de microscopio con texto.
6. **El tipo lo predice el clasificador, no la etiqueta manual**, tanto al ajustar en val como al medir en test, porque en un uso real nadie etiquetaría cada imagen. Las etiquetas manuales entran como **datos de entrenamiento del clasificador** (las 1157 del train). La revisión manual de val y test solo serviría para el análisis por tipo de la presentación.
7. **Clasificador kNN sobre DINOv2** (ViT-S/14, 7 vecinos, voto ponderado por similitud). Reutiliza los embeddings que el equipo ya usaba para ordenar el etiquetado y no hay que entrenar nada. Exactitud en train dejando cada imagen fuera: 0,787 en cuatro tipos.
8. **Todo se elige en validación con mAP@[.5:.95]**, el mismo criterio con el que se eligieron los checkpoints de los baselines. Rejilla: peso de YOLO {0,25; 0,5; 0,75; 1}, umbral de solapamiento {0,5; 0,55; 0,6; 0,7}, confianza mínima {0,001; 0,01; 0,05}. El ensemble por tipo reutiliza el umbral y la confianza mínima del fijo y solo ajusta los dos pesos. Test se evalúa una sola vez con los parámetros ya fijados.
9. **Comparaciones escalonadas con intervalo de confianza** (bootstrap pareado por imágenes de test): ensemble fijo frente a Faster (¿sirve combinar?) y ensemble por tipo frente a fijo (¿sirve usar el tipo de imagen?). Cada paso cambia una sola cosa. Recordatorio: repetir el mismo entrenamiento de Faster varió 0,006 de mAP, así que diferencias menores de unas 0,01 no son concluyentes.
10. **Ablación: WBF aplicado a cada modelo por separado.** Separa cuánto se gana solo por unir las cajas duplicadas de un modelo (sobre todo YOLO) de cuánto se gana por combinar los dos.

## 5. Resultados

Detalle y figuras en [`notebooks/07_ensemble_analysis.ipynb`](../notebooks/07_ensemble_analysis.ipynb) y `results/figures/ensemble/faster_yolo_baseline/`.

**Parámetros elegidos en validación:** peso de YOLO 0,75, umbral de solapamiento 0,7 y confianza mínima 0,05. El ensemble por tipo eligió **el mismo peso (0,75) dentro y fuera de microscopía**, así que es idéntico al fijo.

| Sistema | val mAP@[.5:.95] | test mAP@[.5:.95] | test mAP@0.5 | test mAP@0.75 | test AR@100 | Latencia CPU |
|---|---|---|---|---|---|---|
| Faster R-CNN | 0,663 | 0,623 | 0,784 | 0,745 | 0,749 | 3241 ms |
| YOLOv10-S | 0,612 | 0,537 | 0,683 | 0,638 | 0,699 | 265 ms |
| WBF solo sobre YOLO | 0,607 | — | — | — | — | — |
| **Ensemble (fijo = por tipo)** | **0,686** | **0,635** | **0,789** | **0,756** | **0,767** | 3507 ms |

- **Ensemble frente a Faster en test: +0,012** (IC 95 % [+0,000, +0,026]; probabilidad de no mejorar 0,015). Es el doble del ruido entre entrenamientos, pero el intervalo roza el cero: **mejora real pero pequeña**. En val la ganancia era +0,022, más optimista porque ahí se eligieron los parámetros.
- **Ensemble por tipo frente a fijo: 0** (mismos parámetros).

**De dónde viene la mejora** (desglose TIDE en test, mAP@0.5 perdido; menos es mejor):

| Error | Faster | YOLO | Ensemble |
|---|---|---|---|
| Clasificación | 0,088 | 0,119 | 0,089 |
| Localización | 0,004 | 0,009 | 0,006 |
| Duplicado | 0,000 | 0,011 | 0,000 |
| Fondo | 0,049 | 0,050 | 0,054 |
| No detectado | 0,020 | 0,017 | 0,014 |

### Qué hipótesis se confirman y cuáles no

| Hipótesis (sección 3–4) | Resultado |
|---|---|
| Los modelos se complementan y combinarlos mejora | **Sí**: +0,012 en test, +0,022 en val |
| La fusión corrige la especie (error de clasificación) | **No**: el error de clasificación queda igual (0,088 → 0,089) |
| La fusión reduce los falsos positivos de fondo | **No**: empeora un poco (0,049 → 0,054), porque YOLO añade cajas que Faster no proponía |
| Se encuentran más huevos | **Sí**: no detectados 0,020 → 0,014 (de 51 a 42 huevos sin ninguna detección); AR@100 0,749 → 0,767 |
| Promediar cajas las ajusta mejor | **Sí**: la mejora es mayor con IoU exigente (mAP@0.75 +0,011) que con IoU 0,5 (mAP@0.5 +0,005) |
| Se unen los duplicados de YOLO | **Sí** (384 → 8), pero por sí solo no mejora (WBF solo sobre YOLO: 0,612 → 0,607) |
| El peso de YOLO debe depender del tipo de imagen | **No se confirma**: en val el mejor peso es el mismo en los dos grupos |

Otros resultados:
- **Por especie**, mejora en 6 de 8 (Fasciola +0,035, Enterobius +0,023, Hymenolepis +0,019) y empeora en Ancylostoma (−0,013) y Taenia (−0,007), justo donde YOLO es más flojo (AP 0,527 y 0,364).
- **Por tipo de imagen** (tipo predicho en test), la ganancia es mayor fuera de microscopía (0,508 → 0,537, 90 imágenes) que en microscopía (0,661 → 0,666, 321 imágenes). Al revés de lo que sugería val, y sin intervalo por grupo: solo descriptivo.
- **Latencia**: 3,5 s por imagen ejecutando los dos detectores uno tras otro (+8 % sobre Faster). La fusión cuesta 0,3 ms. Frente a YOLO solo, 13 veces más lento a cambio de +0,098 de mAP.

### Cómo contarlo en la presentación

1. **Hipótesis**: los dos detectores fallan en sitios distintos (YOLO encuentra un 10 % de huevos que Faster no ve) y YOLO rinde peor fuera de microscopía, así que un ensemble que pondere a YOLO según el tipo de imagen debería mejorar.
2. **Método**: WBF propio validado contra la librería de los autores; clasificador de tipo entrenado con nuestras 1157 etiquetas manuales; todo elegido en val y test medido una vez con intervalo de confianza.
3. **Resultado**: el ensemble mejora (+0,012, IC [0,000; 0,026]), sobre todo porque encuentra más huevos y ajusta mejor las cajas, no porque corrija la especie; ponderar por tipo no aporta.
4. **Coste**: +8 % de latencia sobre Faster.
5. **Lección**: medir *de dónde* viene la mejora (TIDE) desmontó nuestra explicación inicial; y el etiquetado manual sirvió para formular y descartar una hipótesis con datos, aunque el sistema final no lo necesite.

## 6. Conclusión de equipo: combinar con A y B

El ensemble solo necesita las predicciones guardadas de cada modelo, así que se aplica igual sobre los modelos mejorados de Rodrigo (A, aumentación) y Lucas (B, copy-paste):

```bash
python scripts/evaluation/evaluate_faster_rcnn.py --run-name <run_mejorado> --split val
python scripts/evaluation/evaluate_faster_rcnn.py --run-name <run_mejorado> --split test
python scripts/evaluation/evaluate_yolo.py --run-name <run_mejorado> --split val
python scripts/evaluation/evaluate_yolo.py --run-name <run_mejorado> --split test
python scripts/evaluation/evaluate_ensemble.py --member fasterrcnn_v2:<run_mejorado> --member yolov10s:<run_mejorado> --name final
```

Las tres mejoras actúan en sitios distintos (A y B en el entrenamiento, D' en la inferencia), así que el sistema final del equipo sería: **los mejores modelos de A y B + ensemble por tipo**.

## 7. Limitaciones

- Val es pequeño (215 imágenes, 49 que no son de microscopía): los pesos por tipo pueden sobreajustarse a val. Por eso solo hay dos grupos y la comparación se hace en test con intervalo de confianza.
- El tipo de val y test es el predicho, no el revisado a mano; el clasificador se equivoca en una de cada cinco imágenes de train (sobre todo diapositivas frente a microscopía).
- La latencia del ensemble es la suma de los dos detectores y la fusión (el clasificador de tipo no hace falta, porque la ponderación por tipo no aportó): 13 veces más lento que YOLO solo.
- Los intervalos de confianza solo recogen la variación por imágenes de test, no la del entrenamiento (unas 0,006 de mAP entre dos entrenamientos idénticos de Faster).

## 8. Reproducir

```bash
python scripts/evaluation/evaluate_ensemble.py
```

Lee las predicciones de `results/metrics/{fasterrcnn_v2,yolov10s}/baseline/` y escribe en `results/metrics/ensemble/faster_yolo_baseline/`: la rejilla de validación (`val_grid.csv`), los parámetros elegidos y los resultados de test con sus intervalos (`summary.json`), el tipo predicho de cada imagen (`image_types.csv`) y, para cada variante, los mismos archivos de métricas y predicciones que los baselines, para analizarlos con los mismos notebooks. La primera ejecución descarga DINOv2 y calcula los embeddings (unos 15 minutos en CPU; quedan en `.cache/`).
