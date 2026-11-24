set shell := ["bash", "-cu"]

# inicia el clasificador visual local del conjunto de entrenamiento
label:
    python scripts/labeling/classify_train_images.py

# descarga los pesos preentrenados de faster rcnn y yolo
weights:
    python scripts/models/download_weights.py

# descarga las particiones originales del dataset desde hugging face
dataset:
    python scripts/data/download_dataset.py

# convierte el dataset descargado a imágenes y anotaciones coco y yolo
prepare:
    python scripts/data/prepare_dataset.py
