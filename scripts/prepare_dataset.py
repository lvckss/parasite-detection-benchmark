#!/usr/bin/env python3
"""Export the original Parquet splits as shared images with COCO and YOLO labels."""

import argparse
import io
import json
from collections import Counter
from pathlib import Path

from download_dataset import CONFIG, RAW, ROOT, sha256


def prepare(raw_dir, output_dir):
    # Import here so --help works before environment setup.
    import pyarrow.parquet as pq
    from PIL import Image

    config = json.loads(CONFIG.read_text())
    categories = [{"id": int(key), "name": value} for key, value in config["classes"].items()]
    label_map = {category["id"]: index for index, category in enumerate(categories)}
    summary = {}
    for split, spec in config["splits"].items():
        source = raw_dir / spec["file"]
        if not source.is_file() or sha256(source) != spec["sha256"]:
            raise ValueError(f"Missing or unexpected {source.name}. Run scripts/download_dataset.py first.")
        table = pq.read_table(source)
        if table.num_rows != spec["images"]:
            raise ValueError(f"Unexpected number of images in {split}: {table.num_rows}")
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        images, annotations, counts = [], [], Counter()
        for row in table.to_pylist():
            image_id = int(row["image_id"])
            content = row["image"]["bytes"]
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
                extension = ".jpg" if image.format == "JPEG" else ".png" if image.format == "PNG" else None
                if extension is None:
                    raise ValueError(f"Unexpected image format: {image.format}")
            filename = f"{image_id:06d}{extension}"
            (image_dir / filename).write_bytes(content)
            images.append({"id": image_id, "file_name": filename, "width": width, "height": height})
            objects = row["objects"]
            if isinstance(objects, dict):
                objects = [dict(zip(objects, values)) for values in zip(*objects.values(), strict=True)]
            labels = []
            for obj in objects:
                category_id = int(obj["category"])
                if category_id not in label_map:
                    raise ValueError(f"Unexpected category {category_id} in {split}, image {image_id}")
                x, y, w, h = map(float, obj["bbox"])
                if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width + 1e-3 or y + h > height + 1e-3:
                    raise ValueError(f"Invalid bounding box {obj['bbox']} in {split}, image {image_id}")
                annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                    "category_id": category_id, "bbox": [x, y, w, h],
                                    "area": w * h, "iscrowd": 0})
                labels.append(f"{label_map[category_id]} {(x + w / 2) / width:.8f} "
                              f"{(y + h / 2) / height:.8f} {w / width:.8f} {h / height:.8f}")
                counts[config["classes"][str(category_id)]] += 1
            (label_dir / f"{image_id:06d}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))
        if len({image["id"] for image in images}) != len(images):
            raise ValueError(f"Duplicate image IDs in {split}")
        annotation_dir = output_dir / "annotations"
        annotation_dir.mkdir(exist_ok=True)
        coco = {"info": {"description": config["repository"], "revision": config["revision"]},
                "images": images, "annotations": annotations, "categories": categories}
        (annotation_dir / f"instances_{split}.json").write_text(json.dumps(coco) + "\n")
        summary[split] = {"images": len(images), "objects": len(annotations), "objects_per_class": dict(counts)}
        print(f"{split}: {len(images)} images, {len(annotations)} objects", flush=True)
    # JSON strings are valid YAML scalars and safely quote paths and class names.
    yaml_lines = [f"path: {json.dumps(str(output_dir.resolve()))}",
                  "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml_lines.extend(f"  {index}: {json.dumps(category['name'])}" for index, category in enumerate(categories))
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n")
    mapping = [{"name": c["name"], "coco_id": c["id"], "torchvision_label": c["id"], "yolo_id": label_map[c["id"]]}
               for c in categories]
    (output_dir / "class_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Prepared dataset: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/converted/parasites-1s07h")
    args = parser.parse_args()
    try:
        prepare(args.raw_dir.resolve(), args.output_dir.resolve())
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")


if __name__ == "__main__":
    main()
