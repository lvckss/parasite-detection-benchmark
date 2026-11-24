#!/usr/bin/env python3
"""Create a reproducible COCO/YOLO dataset with offline training augmentations."""

import argparse
import json
import math
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/converted/parasites-1s07h"
DEFAULT_CONFIG = ROOT / "configs/augmentation.json"
DEFAULT_OUTPUT = ROOT / "data/preprocessed/parasites-1s07h-augmented"


def _selected(specification, rng):
    return specification["enabled"] and rng.random() < specification.get("probability", 1.0)


def _map_points(points, matrix):
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    mapped = homogeneous @ matrix.T
    return mapped[:, :2] / mapped[:, 2:3]


def _box_boundary(box, samples_per_edge=8):
    x, y, width, height = box
    horizontal = np.linspace(x, x + width, samples_per_edge, endpoint=False)
    vertical = np.linspace(y, y + height, samples_per_edge, endpoint=False)
    return np.vstack((
        np.column_stack((horizontal, np.full_like(horizontal, y))),
        np.column_stack((np.full_like(vertical, x + width), vertical)),
        np.column_stack((horizontal[::-1], np.full_like(horizontal, y + height))),
        np.column_stack((np.full_like(vertical, x), vertical[::-1])),
    ))


def _transform_boxes(boxes, matrix, width, height, min_visibility, nonlinear=False):
    transformed = []
    for annotation in boxes:
        original_box = annotation["bbox"]
        if nonlinear:
            points = _box_boundary(original_box)
            mapped = matrix(points)
        else:
            x, y, box_width, box_height = original_box
            corners = np.array([
                [x, y],
                [x + box_width, y],
                [x + box_width, y + box_height],
                [x, y + box_height],
            ])
            mapped = _map_points(corners, matrix)

        x1 = float(mapped[:, 0].min())
        y1 = float(mapped[:, 1].min())
        x2 = float(mapped[:, 0].max())
        y2 = float(mapped[:, 1].max())
        raw_area = max(x2 - x1, 0) * max(y2 - y1, 0)
        x1_clipped = min(max(x1, 0.0), float(width))
        y1_clipped = min(max(y1, 0.0), float(height))
        x2_clipped = min(max(x2, 0.0), float(width))
        y2_clipped = min(max(y2, 0.0), float(height))
        clipped_width = x2_clipped - x1_clipped
        clipped_height = y2_clipped - y1_clipped
        clipped_area = clipped_width * clipped_height
        if (
            clipped_width <= 1
            or clipped_height <= 1
            or raw_area <= 0
            or clipped_area / raw_area < min_visibility
        ):
            continue

        transformed.append({
            "category_id": annotation["category_id"],
            "iscrowd": annotation.get("iscrowd", 0),
            "bbox": [x1_clipped, y1_clipped, clipped_width, clipped_height],
        })
    return transformed


def _apply_affine(image, boxes, matrix, min_visibility):
    height, width = image.shape[:2]
    transformed_image = cv2.warpAffine(
        image,
        matrix[:2].astype(np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    transformed_boxes = _transform_boxes(boxes, matrix, width, height, min_visibility)
    return transformed_image, transformed_boxes


def _apply_perspective(image, boxes, matrix, min_visibility):
    height, width = image.shape[:2]
    transformed_image = cv2.warpPerspective(
        image,
        matrix.astype(np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    transformed_boxes = _transform_boxes(boxes, matrix, width, height, min_visibility)
    return transformed_image, transformed_boxes


def _sample_elastic_field(height, width, alpha, sigma, rng):
    displacement_x = rng.normal(size=(height, width)).astype(np.float32)
    displacement_y = rng.normal(size=(height, width)).astype(np.float32)
    displacement_x = cv2.GaussianBlur(displacement_x, (0, 0), sigma)
    displacement_y = cv2.GaussianBlur(displacement_y, (0, 0), sigma)
    for displacement in (displacement_x, displacement_y):
        magnitude = float(np.max(np.abs(displacement)))
        if magnitude > 0:
            displacement *= alpha / magnitude
    return displacement_x, displacement_y


def _apply_elastic(image, boxes, alpha, sigma, rng, min_visibility):
    height, width = image.shape[:2]
    dx, dy = _sample_elastic_field(height, width, alpha, sigma, rng)
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    transformed_image = cv2.remap(
        image,
        grid_x + dx,
        grid_y + dy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    def inverse_map(points):
        points = points.astype(np.float32)
        mapped = points.copy()
        for _ in range(8):
            sample_x = mapped[:, 0].reshape(1, -1)
            sample_y = mapped[:, 1].reshape(1, -1)
            sampled_dx = cv2.remap(
                dx, sample_x, sample_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).reshape(-1)
            sampled_dy = cv2.remap(
                dy, sample_x, sample_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).reshape(-1)
            mapped[:, 0] = points[:, 0] - sampled_dx
            mapped[:, 1] = points[:, 1] - sampled_dy
        return mapped

    transformed_boxes = _transform_boxes(
        boxes, inverse_map, width, height, min_visibility, nonlinear=True
    )
    return transformed_image, transformed_boxes


def _apply_crop(image, boxes, specification, rng, min_visibility):
    height, width = image.shape[:2]
    minimum, maximum = specification["removed_fraction_per_side"]
    crop_left, crop_right, crop_top, crop_bottom = [
        int(rng.uniform(minimum, maximum) * length)
        for length in (width, width, height, height)
    ]
    x2 = width - crop_right
    y2 = height - crop_bottom
    cropped = image[crop_top:y2, crop_left:x2]
    if cropped.size == 0:
        raise ValueError("Crop configuration removes the whole image.")
    transformed_image = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    matrix = np.array([
        [width / (x2 - crop_left), 0, -crop_left * width / (x2 - crop_left)],
        [0, height / (y2 - crop_top), -crop_top * height / (y2 - crop_top)],
        [0, 0, 1],
    ])
    transformed_boxes = _transform_boxes(boxes, matrix, width, height, min_visibility)
    return transformed_image, transformed_boxes


def _apply_photometric(image, transformations, rng, np_rng):
    image = image.astype(np.float32)
    applied = []
    brightness = transformations["brightness"]
    contrast = transformations["contrast"]
    apply_brightness = _selected(brightness, rng)
    apply_contrast = _selected(contrast, rng)
    if apply_brightness or apply_contrast:
        mean = image.mean(axis=(0, 1), keepdims=True)
        if apply_contrast:
            applied.append("contrast")
            image = (image - mean) * rng.uniform(*contrast["factor"]) + mean
        if apply_brightness:
            applied.append("brightness")
            image *= rng.uniform(*brightness["factor"])

    image = np.clip(image, 0, 255).astype(np.uint8)

    hue_saturation = transformations["hue_saturation"]
    if _selected(hue_saturation, rng):
        applied.append("hue_saturation")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hue_delta = rng.uniform(*hue_saturation["hue_degrees"]) / 2
        saturation_delta = rng.uniform(*hue_saturation["saturation_delta"])
        hsv[:, :, 0] = (hsv[:, :, 0] + int(round(hue_delta))) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + saturation_delta, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    channel_shuffle = transformations["channel_shuffle"]
    if _selected(channel_shuffle, rng):
        applied.append("channel_shuffle")
        image = image[:, :, np_rng.permutation(3)]

    gaussian_noise = transformations["gaussian_noise"]
    if _selected(gaussian_noise, rng):
        applied.append("gaussian_noise")
        low, high = gaussian_noise["standard_deviation_fraction_of_255"]
        sigma = rng.uniform(low, high) * 255
        noise = np_rng.normal(0, sigma, size=image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    gaussian_blur = transformations["gaussian_blur"]
    if _selected(gaussian_blur, rng):
        sigma = rng.uniform(*gaussian_blur["sigma"])
        if sigma > 0:
            applied.append("gaussian_blur")
            image = cv2.GaussianBlur(image, (0, 0), sigma)

    average_blur = transformations["average_blur"]
    if _selected(average_blur, rng):
        applied.append("average_blur")
        low, high = average_blur["kernel_size"]
        candidates = [size for size in range(int(low), int(high) + 1) if size % 2 == 1]
        if not candidates:
            raise ValueError("average_blur.kernel_size must include an odd size.")
        kernel_size = rng.choice(candidates)
        image = cv2.blur(image, (kernel_size, kernel_size))

    sharpen = transformations["sharpen"]
    if _selected(sharpen, rng):
        applied.append("sharpen")
        alpha = rng.uniform(*sharpen["alpha"])
        lightness = rng.uniform(*sharpen["lightness"])
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        image = np.clip(
            image.astype(np.float32) * (1 + alpha * lightness)
            - blurred.astype(np.float32) * (alpha * lightness),
            0,
            255,
        ).astype(np.uint8)

    dropout = transformations["dropout"]
    if _selected(dropout, rng):
        applied.append("dropout")
        height, width = image.shape[:2]
        min_holes, max_holes = dropout["holes"]
        min_fraction, max_fraction = dropout["hole_size_fraction"]
        for _ in range(rng.randint(int(min_holes), int(max_holes))):
            hole_width = max(1, int(rng.uniform(min_fraction, max_fraction) * width))
            hole_height = max(1, int(rng.uniform(min_fraction, max_fraction) * height))
            x = rng.randint(0, max(0, width - hole_width))
            y = rng.randint(0, max(0, height - hole_height))
            fill = np_rng.integers(0, 256, size=(1, 1, 3), dtype=np.uint8)
            image[y:y + hole_height, x:x + hole_width] = fill

    return image, applied


def augment_image(image, boxes, config, seed):
    """Apply configured image transforms and the same geometry to COCO boxes."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    transformations = config["transformations"]
    height, width = image.shape[:2]
    min_visibility = config["output"]["min_box_visibility"]
    applied = []

    selected = {}
    for name in ("horizontal_flip", "vertical_flip", "rotation", "scale", "shear"):
        selected[name] = _selected(transformations[name], rng)

    if any(selected.values()):
        center_x, center_y = width / 2, height / 2
        to_origin = np.array([
            [1, 0, -center_x],
            [0, 1, -center_y],
            [0, 0, 1],
        ], dtype=np.float64)
        from_origin = np.array([
            [1, 0, center_x],
            [0, 1, center_y],
            [0, 0, 1],
        ], dtype=np.float64)
        matrix = np.eye(3, dtype=np.float64)
        if selected["horizontal_flip"]:
            applied.append("horizontal_flip")
            matrix = matrix @ np.diag([-1, 1, 1])
        if selected["vertical_flip"]:
            applied.append("vertical_flip")
            matrix = matrix @ np.diag([1, -1, 1])
        if selected["rotation"]:
            applied.append("rotation")
            angle = math.radians(rng.uniform(*transformations["rotation"]["degrees"]))
            cosine, sine = math.cos(angle), math.sin(angle)
            matrix = matrix @ np.array([
                [cosine, -sine, 0],
                [sine, cosine, 0],
                [0, 0, 1],
            ])
        if selected["scale"]:
            applied.append("scale")
            factor = rng.uniform(*transformations["scale"]["factor"])
            matrix = matrix @ np.diag([factor, factor, 1])
        if selected["shear"]:
            applied.append("shear")
            shear = transformations["shear"]
            shear_x = math.tan(math.radians(rng.uniform(*shear["x_degrees"])))
            shear_y = math.tan(math.radians(rng.uniform(*shear["y_degrees"])))
            matrix = matrix @ np.array([
                [1, shear_x, 0],
                [shear_y, 1, 0],
                [0, 0, 1],
            ])
        matrix = from_origin @ matrix @ to_origin
        image, boxes = _apply_affine(image, boxes, matrix, min_visibility)

    crop = transformations["crop"]
    if _selected(crop, rng):
        applied.append("crop")
        image, boxes = _apply_crop(image, boxes, crop, rng, min_visibility)

    perspective = transformations["perspective"]
    if _selected(perspective, rng):
        applied.append("perspective")
        max_fraction = rng.uniform(*perspective["max_displacement_fraction"])
        corners = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ], dtype=np.float32)
        offsets = np.array([
            [rng.uniform(-max_fraction, max_fraction) * width,
             rng.uniform(-max_fraction, max_fraction) * height]
            for _ in range(4)
        ], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(corners, corners + offsets)
        image, boxes = _apply_perspective(image, boxes, matrix, min_visibility)

    elastic = transformations["elastic"]
    if _selected(elastic, rng):
        applied.append("elastic")
        alpha = rng.uniform(*elastic["alpha"]) if isinstance(elastic["alpha"], list) else elastic["alpha"]
        sigma = rng.uniform(*elastic["sigma"]) if isinstance(elastic["sigma"], list) else elastic["sigma"]
        image, boxes = _apply_elastic(
            image, boxes, alpha, sigma, np_rng, min_visibility
        )

    image, photometric = _apply_photometric(image, transformations, rng, np_rng)
    applied.extend(photometric)
    return image, boxes, applied


def _augment_variant(image, boxes, config, seed, image_id):
    for attempt in range(config["output"]["max_attempts_per_variant"]):
        augmented, augmented_boxes, applied = augment_image(
            image.copy(), boxes, config, seed + attempt
        )
        if boxes and not augmented_boxes:
            continue
        return augmented, augmented_boxes, applied
    raise ValueError(
        f"All boxes were removed in {config['output']['max_attempts_per_variant']} "
        f"augmentation attempts for image {image_id}."
    )


def _load_annotations(coco):
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    return by_image


def _add_coco_annotations(output_annotations, image_id, boxes, next_annotation_id):
    for box in boxes:
        x, y, width, height = box["bbox"]
        output_annotations.append({
            "id": next_annotation_id,
            "image_id": image_id,
            "category_id": int(box["category_id"]),
            "bbox": [float(x), float(y), float(width), float(height)],
            "area": float(width * height),
            "iscrowd": int(box.get("iscrowd", 0)),
        })
        next_annotation_id += 1
    return next_annotation_id


def _write_image(path, image, jpeg_quality):
    parameters = (
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        if path.suffix.lower() in {".jpg", ".jpeg"}
        else []
    )
    success, encoded = cv2.imencode(path.suffix, image, parameters)
    if not success:
        raise OSError(f"Could not encode augmented image: {path}")
    path.write_bytes(encoded.tobytes())


def _read_image(path):
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def _to_box_records(annotations):
    return [
        {
            "category_id": int(annotation["category_id"]),
            "iscrowd": int(annotation.get("iscrowd", 0)),
            "bbox": [float(value) for value in annotation["bbox"]],
        }
        for annotation in annotations
    ]


def _write_split(source_dir, output_dir, split, config, rng_seed):
    annotation_path = source_dir / "annotations" / f"instances_{split}.json"
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations_by_image = _load_annotations(coco)
    image_dir = output_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)

    output_images = []
    output_annotations = []
    annotation_id = 1
    next_image_id = max((image["id"] for image in coco["images"]), default=-1) + 1
    include_originals = split != "train" or config["include_originals"]

    for image_info in coco["images"]:
        source_image = source_dir / "images" / split / image_info["file_name"]
        if not source_image.is_file():
            raise FileNotFoundError(f"Image referenced by COCO is missing: {source_image}")
        source_boxes = _to_box_records(annotations_by_image[image_info["id"]])

        if include_originals:
            shutil.copy2(source_image, image_dir / image_info["file_name"])
            output_images.append(dict(image_info))
            annotation_id = _add_coco_annotations(
                output_annotations, image_info["id"], source_boxes, annotation_id
            )

        if split != "train":
            continue

        image = _read_image(source_image)
        height, width = image.shape[:2]
        if (width, height) != (image_info["width"], image_info["height"]):
            raise ValueError(f"Image dimensions disagree with COCO metadata: {source_image}")

        for variant in range(1, config["variants_per_image"] + 1):
            seed = rng_seed + int(image_info["id"]) * 1009 + variant
            augmented, augmented_boxes, applied = _augment_variant(
                image, source_boxes, config, seed, image_info["id"]
            )

            filename = (
                f"{Path(image_info['file_name']).stem}_aug{variant:02d}"
                f"{Path(image_info['file_name']).suffix}"
            )
            target_path = image_dir / filename
            _write_image(target_path, augmented, config["output"]["jpeg_quality"])

            output_images.append({
                "id": next_image_id,
                "file_name": filename,
                "width": width,
                "height": height,
                "source_image_id": image_info["id"],
                "augmentation_index": variant,
                "applied_transforms": applied,
            })
            annotation_id = _add_coco_annotations(
                output_annotations, next_image_id, augmented_boxes, annotation_id
            )
            next_image_id += 1

    output_annotations_dir = output_dir / "annotations"
    output_annotations_dir.mkdir(parents=True, exist_ok=True)
    output_coco = {
        "info": {
            **coco.get("info", {}),
            "augmentation_config": config["name"],
        },
        "images": output_images,
        "annotations": output_annotations,
        "categories": coco["categories"],
    }
    (output_annotations_dir / f"instances_{split}.json").write_text(
        json.dumps(output_coco, indent=2) + "\n", encoding="utf-8"
    )
    return output_coco


def _write_yolo_labels(output_dir, coco_splits):
    for split, coco in coco_splits.items():
        label_dir = output_dir / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        category_ids = sorted(category["id"] for category in coco["categories"])
        yolo_ids = {category_id: index for index, category_id in enumerate(category_ids)}
        annotations_by_image = _load_annotations(coco)

        for image in coco["images"]:
            width, height = image["width"], image["height"]
            lines = []
            for annotation in annotations_by_image[image["id"]]:
                x, y, box_width, box_height = annotation["bbox"]
                center_x = (x + box_width / 2) / width
                center_y = (y + box_height / 2) / height
                lines.append(
                    f"{yolo_ids[annotation['category_id']]} {center_x:.8f} "
                    f"{center_y:.8f} {box_width / width:.8f} "
                    f"{box_height / height:.8f}"
                )
            label_name = f"{Path(image['file_name']).stem}.txt"
            (label_dir / label_name).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )


def _write_metadata(output_dir, dataset_root, coco_splits, config):
    categories = coco_splits["train"]["categories"]
    category_ids = sorted(category["id"] for category in categories)
    names = {category["id"]: category["name"] for category in categories}
    label_map = {category_id: index for index, category_id in enumerate(category_ids)}

    yaml_lines = [
        f"path: {json.dumps(str(dataset_root.resolve()))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    yaml_lines.extend(
        f"  {label_map[category_id]}: {json.dumps(names[category_id])}"
        for category_id in category_ids
    )
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    mapping = [
        {
            "name": names[category_id],
            "coco_id": category_id,
            "torchvision_label": category_id,
            "yolo_id": label_map[category_id],
        }
        for category_id in category_ids
    ]
    (output_dir / "class_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "augmentation_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    summary = {}
    for split, coco in coco_splits.items():
        counts = Counter(annotation["category_id"] for annotation in coco["annotations"])
        summary[split] = {
            "images": len(coco["images"]),
            "objects": len(coco["annotations"]),
            "objects_per_class": {
                names[category_id]: counts[category_id] for category_id in category_ids
            },
        }
    summary["augmentation"] = {
        "config": config["name"],
        "variants_per_original_train_image": config["variants_per_image"],
        "originals_in_train": config["include_originals"],
        "generated_train_images": sum(
            image.get("augmentation_index") is not None
            for image in coco_splits["train"]["images"]
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def create_augmented_dataset(source_dir, output_dir, config_path):
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["variants_per_image"] < 1:
        raise ValueError("variants_per_image must be at least 1.")
    if config["apply_to"] != ["train"]:
        raise ValueError("Offline augmentation is restricted to the train split.")
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists; choose a new output directory to avoid overwriting it: {output_dir}"
        )
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Converted dataset not found: {source_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        coco_splits = {}
        for split in ("train", "val", "test"):
            coco_splits[split] = _write_split(
                source_dir, staging_dir, split, config, config["seed"]
            )
            print(
                f"{split}: {len(coco_splits[split]['images'])} images, "
                f"{len(coco_splits[split]['annotations'])} objects",
                flush=True,
            )

        _write_yolo_labels(staging_dir, coco_splits)
        _write_metadata(staging_dir, output_dir, coco_splits, config)
        staging_dir.rename(output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    print(f"Augmented dataset created: {output_dir}")


def write_transform_metadata(source_dir, output_dir, config_path):
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    saved_config = json.loads((output_dir / "augmentation_config.json").read_text(encoding="utf-8"))
    if saved_config != config:
        raise ValueError("The existing augmented dataset was created with a different config.")

    source_coco = json.loads(
        (source_dir / "annotations" / "instances_train.json").read_text(encoding="utf-8")
    )
    output_path = output_dir / "annotations" / "instances_train.json"
    output_coco = json.loads(output_path.read_text(encoding="utf-8"))
    source_images = {image["id"]: image for image in source_coco["images"]}
    output_originals = {
        image["id"]: image for image in output_coco["images"] if "source_image_id" not in image
    }
    if {
        image_id: image["file_name"] for image_id, image in output_originals.items()
    } != {
        image_id: image["file_name"] for image_id, image in source_images.items()
    }:
        raise ValueError("The augmented originals do not match the converted training split.")

    variants_by_source = defaultdict(list)
    for image in output_coco["images"]:
        if "source_image_id" not in image:
            continue
        source_id = image["source_image_id"]
        if source_id not in source_images:
            raise ValueError(f"Unknown source_image_id in augmented dataset: {source_id}")
        variants_by_source[source_id].append(image)

    if len(variants_by_source) != len(source_images):
        raise ValueError("The augmented dataset does not contain variants for every training image.")

    annotations_by_image = _load_annotations(source_coco)
    total = sum(len(variants) for variants in variants_by_source.values())
    completed = 0
    for source_id, source_info in source_images.items():
        variants = sorted(
            variants_by_source[source_id], key=lambda image: image["augmentation_index"]
        )
        expected_indices = list(range(1, config["variants_per_image"] + 1))
        if [image.get("augmentation_index") for image in variants] != expected_indices:
            raise ValueError(f"Unexpected augmentation indices for source image {source_id}.")

        source_path = source_dir / "images" / "train" / source_info["file_name"]
        image = _read_image(source_path)
        source_boxes = _to_box_records(annotations_by_image[source_id])
        for variant in variants:
            expected_name = (
                f"{Path(source_info['file_name']).stem}_aug"
                f"{variant['augmentation_index']:02d}{Path(source_info['file_name']).suffix}"
            )
            if variant["file_name"] != expected_name:
                raise ValueError(f"Unexpected augmented filename: {variant['file_name']}")
            augmented_path = output_dir / "images" / "train" / variant["file_name"]
            if not augmented_path.is_file():
                raise FileNotFoundError(f"Augmented image is missing: {augmented_path}")

            seed = (
                config["seed"]
                + int(source_id) * 1009
                + variant["augmentation_index"]
            )
            _, _, applied = _augment_variant(image, source_boxes, config, seed, source_id)
            variant["applied_transforms"] = applied
            completed += 1
            if completed % 250 == 0 or completed == total:
                print(f"Transform metadata: {completed}/{total}", flush=True)

    temporary_path = output_path.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(output_coco, indent=2) + "\n", encoding="utf-8"
        )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"Transform metadata saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--write-transform-metadata",
        action="store_true",
        help="Add actual transform names to an existing augmented dataset without rewriting images.",
    )
    args = parser.parse_args()
    try:
        if args.write_transform_metadata:
            write_transform_metadata(args.source_dir, args.output_dir, args.config)
        else:
            create_augmented_dataset(args.source_dir, args.output_dir, args.config)
    except (OSError, ValueError, ImportError, KeyError) as error:
        parser.exit(1, f"Dataset generation failed: {error}\n")


if __name__ == "__main__":
    main()
