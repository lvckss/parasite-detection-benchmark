set shell := ["bash", "-cu"]

# Inicia el clasificador visual local del split train.
label:
    python scripts/classify_train_images.py

# Descarga los pesos COCO de Faster R-CNN v2 y YOLOv10-S.
weights:
    python scripts/download_weights.py
