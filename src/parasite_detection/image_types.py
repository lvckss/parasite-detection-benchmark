"""tipo de imagen (microscopía, diapositiva, collage, ilustración) a partir de las etiquetas manuales del train

un clasificador knn sobre embeddings dinov2 vit-s/14 aprende de las 1157 etiquetas manuales y predice el tipo
de cualquier imagen. el ensemble lo usa para elegir los pesos de la fusión según el grupo de la imagen
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from parasite_detection.paths import DEDUP_DATA, ROOT

LABEL_FILES = [ROOT / "results/visual-labels/train_dedup_labels.json", ROOT / "results/visual-labels/rodrigo.json"]
CACHE = ROOT / ".cache/visual-labels"
TYPES = ["microscopy", "slide", "collage", "illustration"]
K_NEIGHBORS = 7
IMAGE_SIZE = 518


def manual_train_labels():
    """une los dos archivos de etiquetas del train; cada imagen la etiquetó una persona o dos sin desacuerdo"""
    labels = {}
    for path in LABEL_FILES:
        for item in json.loads(path.read_text(encoding="utf-8"))["items"]:
            category = item["category"]
            if category and category != "uncertain":
                if labels.get(item["id"], category) != category:
                    raise ValueError(f"Etiquetas contradictorias para {item['id']}")
                labels[item["id"]] = category
    return labels


def group_of(image_type):
    # val es pequeño (49 imágenes no son de microscopía), así que solo distinguimos dos grupos
    return "microscopy" if image_type == "microscopy" else "other"


def extract_embeddings(paths, device="cpu", batch_size=16):
    """token cls de dinov2 vit-s/14 con el mismo preprocesado que order_train_images_by_similarity.py"""
    import torch
    from PIL import Image, ImageOps

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True).eval().to(device)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    features = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    # conservamos toda la imagen sobre un lienzo gris cuadrado
                    image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
                    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (127, 127, 127))
                    canvas.paste(image, ((IMAGE_SIZE - image.width) // 2, (IMAGE_SIZE - image.height) // 2))
                    tensors.append(torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).permute(2, 0, 1))
            batch = (torch.stack(tensors).to(device) - mean) / std
            output = model.forward_features(batch)["x_norm_clstoken"]
            features.append(torch.nn.functional.normalize(output.float(), dim=1).cpu())
            print(f"\rEmbeddings: {min(start + batch_size, len(paths))}/{len(paths)}", end="", flush=True)
    print()
    return torch.cat(features).numpy()


def split_embeddings(split, data_dir=DEDUP_DATA, device="cpu"):
    """embeddings de un split, guardados en .cache para no recalcularlos"""
    cache = CACHE / f"dinov2_{split}_embeddings.npz"
    if not cache.is_file():
        coco = json.loads((Path(data_dir) / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))
        records = sorted(coco["images"], key=lambda r: r["id"])
        paths = [Path(data_dir) / "images" / split / r["file_name"] for r in records]
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ids=np.array([r["id"] for r in records]),
                            stems=np.array([p.stem for p in paths]), features=extract_embeddings(paths, device))
    data = np.load(cache)
    return data["ids"], data["stems"], data["features"]


def knn_predict(features, reference, reference_labels, k=K_NEIGHBORS, exclude_self=False):
    """voto de los k vecinos más parecidos (coseno), ponderado por la similitud; devuelve tipo y confianza"""
    similarity = features @ reference.T
    if exclude_self:
        np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argsort(-similarity, axis=1)[:, :k]
    types, confidence = [], []
    for row, indices in zip(similarity, neighbors):
        votes = Counter()
        for j in indices:
            votes[reference_labels[j]] += max(float(row[j]), 0.0)
        label, score = votes.most_common(1)[0]
        types.append(label)
        confidence.append(score / max(sum(votes.values()), 1e-9))
    return np.array(types), np.array(confidence)


def predict_split_types(split, data_dir=DEDUP_DATA, device="cpu"):
    """tipo predicho para cada imagen de un split: {image_id: (tipo, confianza)}"""
    labels = manual_train_labels()
    _, train_stems, train_features = split_embeddings("train", data_dir, device)
    keep = np.array([stem in labels for stem in train_stems])
    reference, reference_labels = train_features[keep], np.array([labels[s] for s in train_stems[keep]])
    ids, _, features = split_embeddings(split, data_dir, device)
    types, confidence = knn_predict(features, reference, reference_labels)
    return {int(i): (str(t), float(c)) for i, t, c in zip(ids, types, confidence)}


def leave_one_out_accuracy(data_dir=DEDUP_DATA, device="cpu"):
    """exactitud del clasificador en el train dejando cada imagen fuera de sus propios vecinos"""
    labels = manual_train_labels()
    _, stems, features = split_embeddings("train", data_dir, device)
    keep = np.array([stem in labels for stem in stems])
    truth = np.array([labels[s] for s in stems[keep]])
    predicted, _ = knn_predict(features[keep], features[keep], truth, exclude_self=True)
    grouped = np.mean([group_of(p) == group_of(t) for p, t in zip(predicted, truth)])
    return float(np.mean(predicted == truth)), float(grouped), truth, predicted
