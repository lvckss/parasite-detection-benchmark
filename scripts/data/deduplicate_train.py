#!/usr/bin/env python3
"""Copy the converted dataset without the train images that nearly duplicate a val or test image."""

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from download_dataset import ROOT

# partimos del dataset convertido y escribimos una variante independiente
SOURCE = ROOT / "data/converted/parasites-1s07h"
OUTPUT = ROOT / "data/preprocessed/parasites-1s07h-dedup"
REMOVED_CSV = ROOT / "results/metrics/deduplication/train_removed_near_duplicates.csv"
SPLITS = ["train", "val", "test"]


def deduplicate(source, output, removed_csv, threshold):
    # cargamos las dependencias al ejecutar la limpieza para mantener accesible la ayuda
    import numpy as np
    from parasite_detection.duplicates import SIZE, correlation, fingerprint

    # las anotaciones nos dan la lista de imágenes de cada partición
    coco = {split: json.loads((source / "annotations" / f"instances_{split}.json").read_text())
            for split in SPLITS}
    # resumimos cada imagen con una miniatura en gris normalizada
    features = {split: np.stack([fingerprint(source / "images" / split / image["file_name"])
                                 for image in coco[split]["images"]]) for split in SPLITS}

    # comparamos cada imagen de train con todas las de validación y test
    eval_images = [(split, image) for split in ["val", "test"] for image in coco[split]["images"]]
    similarity = correlation(np.vstack([features["val"], features["test"]]), features["train"])
    removed = []
    # retiramos cualquier imagen de train con al menos una coincidencia sobre el umbral
    for column in np.flatnonzero((similarity >= threshold).any(axis=0)):
        matches = np.flatnonzero(similarity[:, column] >= threshold)
        # anotamos la coincidencia más fuerte y el número total de coincidencias
        best = matches[similarity[matches, column].argmax()]
        split, image = eval_images[best]
        train_image = coco["train"]["images"][column]
        removed.append({"image_id": train_image["id"], "file_name": train_image["file_name"],
                        "eval_split": split, "eval_file_name": image["file_name"],
                        "similarity": round(float(similarity[best, column]), 6), "eval_matches": len(matches)})
    removed_ids = {row["image_id"] for row in removed}

    # el marcador identifica las carpetas creadas por este script
    marker = output / "deduplication.json"
    if output.exists() and any(output.iterdir()) and not marker.exists():
        raise ValueError(f"{output} is not empty and was not created by this script")
    shutil.rmtree(output, ignore_errors=True)
    (output / "annotations").mkdir(parents=True)
    categories = {category["id"]: category["name"] for category in coco["train"]["categories"]}
    summary = {}
    # filtramos train y copiamos completas las particiones de validación y test
    for split in SPLITS:
        images = [image for image in coco[split]["images"] if image["id"] not in removed_ids or split != "train"]
        kept = {image["id"] for image in images}
        # conservamos las cajas de las imágenes que siguen en la partición
        annotations = [ann for ann in coco[split]["annotations"] if ann["image_id"] in kept]
        for folder in ["images", "labels"]:
            (output / folder / split).mkdir(parents=True)
        for image in images:
            stem = Path(image["file_name"]).stem
            shutil.copy2(source / "images" / split / image["file_name"], output / "images" / split)
            shutil.copy2(source / "labels" / split / f"{stem}.txt", output / "labels" / split)
        (output / "annotations" / f"instances_{split}.json").write_text(
            json.dumps({**coco[split], "images": images, "annotations": annotations}) + "\n")
        counts = Counter(categories[ann["category_id"]] for ann in annotations)
        summary[split] = {"images": len(images), "objects": len(annotations), "objects_per_class": dict(counts)}
        print(f"{split}: {len(images)} images, {len(annotations)} objects", flush=True)

    # adaptamos la ruta de yolo para que apunte a la nueva variante
    yaml = (source / "dataset.yaml").read_text().splitlines()
    yaml[0] = f"path: {json.dumps(str(output.resolve()))}"
    (output / "dataset.yaml").write_text("\n".join(yaml) + "\n")
    shutil.copy2(source / "class_mapping.json", output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    marker.write_text(json.dumps({"source": str(source.relative_to(ROOT)), "fingerprint": f"{SIZE}x{SIZE} gray",
                                  "threshold": threshold, "compared_with": ["val", "test"],
                                  "train_images_removed": len(removed)}, indent=2) + "\n")

    # guardamos el registro de imágenes retiradas para revisar la limpieza
    removed_csv.parent.mkdir(parents=True, exist_ok=True)
    with removed_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(removed[0]) if removed else ["image_id"])
        writer.writeheader()
        writer.writerows(sorted(removed, key=lambda row: row["image_id"]))
    print(f"Removed {len(removed)} train images; list saved to {removed_csv.relative_to(ROOT)}")
    print(f"Deduplicated dataset: {output}")


def main():
    # el umbral controla cuánta similitud exigimos para considerar una coincidencia
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--removed-csv", type=Path, default=REMOVED_CSV)
    parser.add_argument("--threshold", type=float, default=0.90, help="minimum correlation (default: 0.90, as in the notebook)")
    args = parser.parse_args()
    try:
        deduplicate(args.source.resolve(), args.output.resolve(), args.removed_csv.resolve(), args.threshold)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"Deduplication failed: {error}\n")


if __name__ == "__main__":
    main()
