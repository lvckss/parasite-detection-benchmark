set shell := ["bash", "-cu"]

# Inicia el clasificador visual local del split train.
label:
    python scripts/classify_train_images.py
