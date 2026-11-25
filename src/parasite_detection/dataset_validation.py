"""Validation shared by the COCO and YOLO training pipelines."""

import json
import math
import re
from collections import defaultdict
from pathlib import Path


COCO_CATEGORY_IDS = set(range(1, 9))
YOLO_CATEGORY_IDS = set(range(8))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _read_coco_split(data_dir, split):
    annotation_path = Path(data_dir) / "annotations" / f"instances_{split}.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Faltan las anotaciones COCO de {split}: {annotation_path}")
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))

    categories = coco.get("categories", [])
    if len(categories) != len(COCO_CATEGORY_IDS) or {category["id"] for category in categories} != COCO_CATEGORY_IDS:
        raise ValueError(f"Las categorías COCO de {split} deben usar los IDs 1–8.")

    images = coco.get("images", [])
    image_ids = [image["id"] for image in images]
    filenames = [image["file_name"] for image in images]
    if len(set(image_ids)) != len(images) or len(set(filenames)) != len(images):
        raise ValueError(f"IDs y nombres de imagen deben ser únicos en {split}.")
    if len({Path(filename).stem for filename in filenames}) != len(filenames):
        raise ValueError(f"Los nombres de imagen deben tener stems únicos en {split}.")
    if any(image["width"] < 1 or image["height"] < 1 for image in images):
        raise ValueError(f"Las dimensiones de las imágenes de {split} deben ser positivas.")

    image_dir = Path(data_dir) / "images" / split
    expected_files = set(filenames)
    actual_files = {
        path.name for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    } if image_dir.is_dir() else set()
    if actual_files != expected_files:
        raise ValueError(f"Las imágenes de {split} no coinciden con instances_{split}.json.")

    images_by_id = {image["id"]: image for image in images}
    annotations_by_image = defaultdict(list)
    annotation_ids = set()
    for annotation in coco.get("annotations", []):
        if annotation["id"] in annotation_ids:
            raise ValueError(f"IDs de anotación duplicados en {split}: {annotation['id']}.")
        annotation_ids.add(annotation["id"])
        image = images_by_id.get(annotation["image_id"])
        if image is None:
            raise ValueError(f"Anotación {annotation['id']} referencia una imagen inexistente.")
        if annotation["category_id"] not in COCO_CATEGORY_IDS:
            raise ValueError(f"Anotación {annotation['id']} tiene una clase fuera de 1–8.")
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
            raise ValueError(f"Anotación {annotation['id']} tiene una bbox inválida.")
        x, y, width, height = map(float, bbox)
        if (
            x < 0 or y < 0 or width <= 0 or height <= 0
            or x + width > image["width"] + 1e-3
            or y + height > image["height"] + 1e-3
        ):
            raise ValueError(f"Anotación {annotation['id']} tiene una bbox fuera de la imagen.")
        annotations_by_image[annotation["image_id"]].append(annotation)

    return coco, annotations_by_image


def validate_coco_dataset(data_dir, splits=("train", "val"), expected_train_stems=None):
    """Check that COCO annotations and image files are consistent."""
    data_dir = Path(data_dir)
    train = None
    for split in splits:
        coco, _ = _read_coco_split(data_dir, split)
        if split == "train":
            train = coco
    if train is not None and expected_train_stems is not None:
        _validate_train_membership(data_dir, train, expected_train_stems)


def validate_loaded_image_paths(data_dir, split, loaded_paths):
    """Ensure a training framework did not drop or invent annotated images."""
    annotation_path = Path(data_dir) / "annotations" / f"instances_{split}.json"
    expected = {
        image["file_name"]
        for image in json.loads(annotation_path.read_text(encoding="utf-8"))["images"]
    }
    loaded_paths = list(loaded_paths)
    actual = {Path(path).name for path in loaded_paths}
    if len(loaded_paths) != len(expected) or actual != expected:
        raise ValueError(f"El dataloader debe conservar todas las imágenes anotadas de {split}.")


def _validate_train_membership(data_dir, train, expected_train_stems):
    expected = {Path(str(stem)).stem for stem in expected_train_stems}
    images = train["images"]
    augmentation_path = Path(data_dir) / "augmentation_config.json"
    if not augmentation_path.is_file():
        actual = {Path(image["file_name"]).stem for image in images}
        if actual != expected:
            raise ValueError(
                "El train no coincide con el manifiesto deduplicado. "
                "Para usar otra variante, genera sus metadatos de aumentación."
            )
        return

    config = json.loads(augmentation_path.read_text(encoding="utf-8"))
    variants_per_image = config.get("variants_per_image")
    if (
        config.get("apply_to") != ["train"]
        or not isinstance(variants_per_image, int)
        or variants_per_image < 1
        or not isinstance(config.get("include_originals"), bool)
    ):
        raise ValueError("La configuración de aumentación no describe una variante de train válida.")

    originals = [image for image in images if "source_image_id" not in image]
    variants = [image for image in images if "source_image_id" in image]
    if any("augmentation_index" in image for image in originals):
        raise ValueError("Una imagen original no puede tener augmentation_index.")

    include_originals = config["include_originals"]
    if include_originals:
        if {Path(image["file_name"]).stem for image in originals} != expected:
            raise ValueError("Las imágenes originales de train no coinciden con el manifiesto.")
        source_by_id = {image["id"]: image for image in originals}
    elif originals:
        raise ValueError("La configuración excluye originales, pero el train los contiene.")

    variants_by_source = defaultdict(list)
    source_stems = {}
    source_ids_by_stem = {}
    for image in variants:
        source_id = image["source_image_id"]
        index = image.get("augmentation_index")
        if not isinstance(index, int) or not 1 <= index <= variants_per_image:
            raise ValueError(f"Índice de aumentación inválido para {image['file_name']}.")
        if include_originals:
            source = source_by_id.get(source_id)
            if source is None:
                raise ValueError(f"{image['file_name']} referencia una imagen fuente inexistente.")
            expected_name = (
                f"{Path(source['file_name']).stem}_aug{index:02d}"
                f"{Path(source['file_name']).suffix}"
            )
            if image["file_name"] != expected_name:
                raise ValueError(f"Nombre de variante inesperado: {image['file_name']}.")
            source_stem = Path(source["file_name"]).stem
        else:
            match = re.fullmatch(r"(.+)_aug\d+", Path(image["file_name"]).stem)
            if match is None:
                raise ValueError(f"No se puede identificar la imagen fuente de {image['file_name']}.")
            source_stem = match.group(1)
            previous_id = source_ids_by_stem.setdefault(source_stem, source_id)
            if previous_id != source_id:
                raise ValueError(f"El stem {source_stem} apunta a varias imágenes fuente.")
        previous_stem = source_stems.setdefault(source_id, source_stem)
        if previous_stem != source_stem:
            raise ValueError(f"La imagen fuente {source_id} tiene nombres de variante inconsistentes.")
        variants_by_source[source_id].append(index)

    if not include_originals and set(source_stems.values()) != expected:
        raise ValueError("Las variantes no cubren exactamente las imágenes del manifiesto.")
    if not variants:
        raise ValueError("La configuración de aumentación está presente, pero no hay variantes.")

    expected_indices = list(range(1, variants_per_image + 1))
    for source_id, indices in variants_by_source.items():
        if sorted(indices) != expected_indices:
            raise ValueError(f"Faltan o se repiten variantes para la imagen fuente {source_id}.")
    if set(variants_by_source) != set(source_stems):
        raise ValueError("Las variantes no tienen una correspondencia única con sus fuentes.")
    if include_originals and set(variants_by_source) != set(source_by_id):
        raise ValueError("Cada imagen original debe tener todas sus variantes.")


def validate_yolo_dataset(data_dir, expected_train_stems=None, splits=("train", "val", "test")):
    """Validate COCO/YOLO files and either the baseline or generated train variants."""
    data_dir = Path(data_dir)
    coco_splits = {}
    for split in splits:
        coco_splits[split], annotations_by_image = _read_coco_split(data_dir, split)
        coco_splits[split]["_annotations_by_image"] = annotations_by_image

    mapping = json.loads((data_dir / "class_mapping.json").read_text(encoding="utf-8"))
    coco_to_yolo = {}
    first_coco = next(iter(coco_splits.values()))
    coco_names = {
        category["id"]: category["name"]
        for category in first_coco["categories"]
    }
    for split, coco in coco_splits.items():
        if {
            category["id"]: category["name"] for category in coco["categories"]
        } != coco_names:
            raise ValueError(f"Los nombres de clase de {split} no coinciden con el resto del dataset.")
    for item in mapping:
        if (
            item.get("coco_id") not in COCO_CATEGORY_IDS
            or item.get("yolo_id") not in YOLO_CATEGORY_IDS
            or item.get("torchvision_label") != item.get("coco_id")
            or item.get("name") != coco_names.get(item.get("coco_id"))
        ):
            raise ValueError("class_mapping.json debe mapear las mismas clases COCO/torchvision 1–8 a YOLO 0–7.")
        coco_to_yolo[item["coco_id"]] = item["yolo_id"]
    if (
        len(mapping) != len(COCO_CATEGORY_IDS)
        or set(coco_to_yolo) != COCO_CATEGORY_IDS
        or set(coco_to_yolo.values()) != YOLO_CATEGORY_IDS
    ):
        raise ValueError("class_mapping.json debe contener una correspondencia completa de 8 clases.")

    for split, coco in coco_splits.items():
        label_dir = data_dir / "labels" / split
        expected_labels = {f"{Path(image['file_name']).stem}.txt" for image in coco["images"]}
        actual_labels = {path.name for path in label_dir.glob("*.txt")} if label_dir.is_dir() else set()
        if actual_labels != expected_labels:
            raise ValueError(f"Las etiquetas YOLO de {split} no coinciden con las imágenes COCO.")

        for image in coco["images"]:
            label_path = label_dir / f"{Path(image['file_name']).stem}.txt"
            actual_rows = []
            text = label_path.read_text(encoding="utf-8").strip()
            for line_number, line in enumerate(text.splitlines(), start=1) if text else []:
                values = line.split()
                if len(values) != 5:
                    raise ValueError(f"Etiqueta inválida en {label_path}:{line_number}.")
                try:
                    category_id = int(values[0])
                    box = tuple(float(value) for value in values[1:])
                except ValueError as error:
                    raise ValueError(f"Etiqueta inválida en {label_path}:{line_number}.") from error
                if (
                    category_id not in YOLO_CATEGORY_IDS
                    or not all(math.isfinite(value) for value in box)
                    or not 0 <= box[0] <= 1 or not 0 <= box[1] <= 1
                    or not 0 < box[2] <= 1 or not 0 < box[3] <= 1
                ):
                    raise ValueError(f"Etiqueta fuera de rango en {label_path}:{line_number}.")
                actual_rows.append((category_id, box))

            expected_rows = []
            for annotation in coco["_annotations_by_image"].get(image["id"], []):
                x, y, width, height = annotation["bbox"]
                expected_rows.append((
                    coco_to_yolo[annotation["category_id"]],
                    (
                        (x + width / 2) / image["width"],
                        (y + height / 2) / image["height"],
                        width / image["width"],
                        height / image["height"],
                    ),
                ))
            if len(actual_rows) != len(expected_rows):
                raise ValueError(f"El número de cajas no coincide en {label_path}.")
            unmatched = list(expected_rows)
            for actual_category, actual_box in actual_rows:
                match_index = next((
                    index for index, (expected_category, expected_box) in enumerate(unmatched)
                    if actual_category == expected_category
                    and all(abs(actual - expected) <= 1e-5 for actual, expected in zip(actual_box, expected_box))
                ), None)
                if match_index is None:
                    raise ValueError(f"Las cajas YOLO y COCO no coinciden en {label_path}.")
                unmatched.pop(match_index)

    if "train" in coco_splits and expected_train_stems is not None:
        _validate_train_membership(data_dir, coco_splits["train"], expected_train_stems)
