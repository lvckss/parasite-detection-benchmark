set shell := ["bash", "-cu"]

# Inicia el clasificador visual local del split train.
label:
    python scripts/classify_train_images.py

# Descarga los pesos COCO de Faster R-CNN v2 y YOLOv10-S.
weights:
    python scripts/download_weights.py

# Descarga los splits originales del dataset desde Hugging Face.
dataset:
    python scripts/download_dataset.py

# Convierte el dataset descargado a imágenes y anotaciones COCO/YOLO.
prepare:
    python scripts/prepare_dataset.py
