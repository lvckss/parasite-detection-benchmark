#!/usr/bin/env python3
"""Ordena el train deduplicado con DINOv2 ViT-S/14 preentrenado.

El manifiesto reparte cada vecindad visual entre las tres personas para que
todas recorran las vecindades en el mismo orden.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# fijamos las rutas de imágenes y resultados a partir de la raíz del proyecto
ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "data/preprocessed/parasites-1s07h-dedup/images/train"
EMBEDDING_FILE = ROOT / ".cache/visual-labels/dinov2_train_embeddings.npz"
ORDER_FILE = ROOT / "results/visual-labels/dinov2_train_order.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
# el reparto usa siempre el mismo orden de personas
PEOPLE = ["Alejandro", "Rodrigo", "Lucas"]
EXPECTED_IMAGES = 1157


def image_paths() -> list[Path]:
    # comprobamos que existe la variante depurada antes de ordenar las imágenes
    if not IMAGE_DIR.is_dir():
        raise SystemExit(
            f"No se encuentra el train deduplicado: {IMAGE_DIR}\n"
            "Ejecuta python scripts/data/prepare_dataset.py y "
            "python scripts/data/deduplicate_train.py primero."
        )
    # ordenamos los nombres para que la caché pueda identificar el mismo conjunto
    paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(paths) != EXPECTED_IMAGES:
        raise SystemExit(
            f"Se esperaban {EXPECTED_IMAGES} imágenes deduplicadas y se encontraron {len(paths)}. "
            "No se generó ninguna ordenación."
        )
    return paths


def extract_embeddings(paths: list[Path], device_name: str, batch_size: int):
    # importamos las dependencias pesadas solo cuando hay que extraer los vectores
    try:
        import numpy as np
        import torch
        from PIL import Image, ImageOps
    except ImportError as error:
        raise SystemExit(
            "Faltan dependencias. Activa .venv y ejecuta el entorno con PyTorch instalado."
        ) from error

    # comprobamos el dispositivo solicitado antes de cargar el modelo
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA no está disponible en este entorno. Activa el entorno cu128 en una máquina "
            "con la RTX 3080, o usa --device cpu explícitamente."
        )
    device = torch.device("cuda" if device_name == "cuda" else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Extrayendo embeddings en CPU.")

    # torch hub descarga el modelo la primera vez y después reutiliza su caché
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    except Exception as error:
        raise SystemExit(
            "No se pudo cargar DINOv2 desde Torch Hub. La primera ejecución necesita "
            "descargar el código y los pesos preentrenados."
        ) from error
    model.eval().to(device)

    # normalizamos los canales con los valores usados por el modelo preentrenado
    mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    all_features = []
    # extraemos los vectores por lotes con los gradientes desactivados
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    # conservamos toda la imagen y rellenamos el espacio libre del lienzo
                    image.thumbnail((518, 518), Image.Resampling.LANCZOS)
                    canvas = Image.new("RGB", (518, 518), (127, 127, 127))
                    canvas.paste(image, ((518 - image.width) // 2, (518 - image.height) // 2))
                    array = np.asarray(canvas, dtype=np.float32) / 255.0
                    tensors.append(torch.from_numpy(array).permute(2, 0, 1))
            batch = torch.stack(tensors).to(device)
            batch = (batch - mean) / std
            # el token cls resume la información visual de cada imagen
            features = model.forward_features(batch)["x_norm_clstoken"]
            features = torch.nn.functional.normalize(features.float(), dim=1)
            all_features.append(features.cpu())
            done = min(start + len(tensors), len(paths))
            print(f"Embeddings: {done}/{len(paths)}", end="\r", flush=True)
    print()
    return torch.cat(all_features).numpy()


def nearest_neighbor_order(features):
    """Construye un recorrido voraz determinista según similitud coseno."""
    import numpy as np

    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    # con vectores normalizados el producto escalar equivale a la similitud coseno
    similarities = features @ features.T
    np.fill_diagonal(similarities, -np.inf)
    # empezamos por el vector más cercano al centroide para estabilizar el inicio
    centroid = features.mean(axis=0)
    current = int(np.argmax(features @ centroid))
    visited = np.zeros(len(features), dtype=bool)
    order = []
    for _ in range(len(features)):
        order.append(current)
        visited[current] = True
        candidates = similarities[current].copy()
        # elegimos la imagen más parecida entre las que quedan por visitar
        candidates[visited] = -np.inf
        if len(order) < len(features):
            current = int(np.argmax(candidates))
    return order


def make_manifest(paths: list[Path], order: list[int], chunk_size: int) -> dict:
    # dividimos el recorrido en vecindades que se reparten entre las tres personas
    groups = [order[start : start + chunk_size] for start in range(0, len(order), chunk_size)]
    batches = {person: [] for person in PEOPLE}
    counts = {person: 0 for person in PEOPLE}
    group_records = []

    for group_index, group in enumerate(groups):
        quotas = {person: len(group) // len(PEOPLE) for person in PEOPLE}
        remainder = len(group) % len(PEOPLE)
        # asignamos los sobrantes a los lotes más pequeños y resolvemos los empates por orden
        for person in sorted(PEOPLE, key=lambda name: (counts[name], PEOPLE.index(name)))[:remainder]:
            quotas[person] += 1
        cursor = 0
        group_record = {"id": group_index, "images": [paths[i].stem for i in group], "batches": {}}
        # cada persona recibe un tramo de la vecindad sin repetir imágenes
        for person in PEOPLE:
            indices = group[cursor : cursor + quotas[person]]
            cursor += quotas[person]
            ids = [paths[i].stem for i in indices]
            batches[person].extend(ids)
            counts[person] += len(ids)
            group_record["batches"][person] = ids
        group_records.append(group_record)

    ordered_ids = [paths[i].stem for i in order]
    # guardamos tanto el recorrido global como las imágenes asignadas a cada persona
    return {
        "dataset": "Francesco/parasites-1s07h",
        "variant": "deduplicated",
        "split": "train",
        "images": len(paths),
        "model": "DINOv2 ViT-S/14 pretrained",
        "embedding": "normalized CLS token; cosine similarity",
        "ordering": "greedy nearest-neighbor path",
        "neighborhood_size": chunk_size,
        "people": PEOPLE,
        "batch_sizes": counts,
        "ordered_ids": ordered_ids,
        "batches": batches,
        "groups": group_records,
    }


def main() -> None:
    # los argumentos controlan el dispositivo y el tamaño de los lotes y vecindades
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--neighborhood-size", type=int, default=12)
    parser.add_argument("--force", action="store_true", help="Recalcular embeddings guardados")
    args = parser.parse_args()
    if args.batch_size < 1 or args.neighborhood_size < 3:
        parser.error("batch-size debe ser positivo y neighborhood-size al menos 3")

    paths = image_paths()
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit("Falta numpy; activa .venv.") from error

    ids = [path.stem for path in paths]
    # reutilizamos los vectores si pertenecen exactamente a las mismas imágenes
    if EMBEDDING_FILE.exists() and not args.force:
        saved = np.load(EMBEDDING_FILE, allow_pickle=False)
        if saved["ids"].tolist() != ids:
            raise SystemExit("La caché de embeddings no coincide con el split deduplicado actual; usa --force.")
        features = saved["features"]
        print(f"Usando embeddings guardados: {EMBEDDING_FILE.relative_to(ROOT)}")
    else:
        features = extract_embeddings(paths, args.device, args.batch_size)
        EMBEDDING_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(EMBEDDING_FILE, ids=np.asarray(ids), features=features)

    # construimos el recorrido y guardamos el reparto que leerá el clasificador
    order = nearest_neighbor_order(features)
    manifest = make_manifest(paths, order, args.neighborhood_size)
    ORDER_FILE.parent.mkdir(parents=True, exist_ok=True)
    ORDER_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Orden guardado: {ORDER_FILE.relative_to(ROOT)}")
    print("Lotes: " + ", ".join(f"{person}={size}" for person, size in manifest["batch_sizes"].items()))


if __name__ == "__main__":
    main()
