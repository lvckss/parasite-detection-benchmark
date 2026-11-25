import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from parasite_detection.dataset_validation import (
    validate_coco_dataset,
    validate_loaded_image_paths,
    validate_yolo_dataset,
)
from scripts.create_augmented_dataset import create_augmented_dataset


class DatasetValidationTests(unittest.TestCase):
    def _create_dataset(self, root, augmented=False, include_originals=True):
        root.mkdir(parents=True, exist_ok=True)
        categories = [{"id": index, "name": f"class_{index}"} for index in range(1, 9)]
        mapping = [
            {"name": f"class_{index}", "coco_id": index, "torchvision_label": index, "yolo_id": index - 1}
            for index in range(1, 9)
        ]
        (root / "class_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")

        for split, image_id in (("train", 10), ("val", 20), ("test", 30)):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            (root / "annotations").mkdir(exist_ok=True)
            image_records = []
            if split != "train" or include_originals:
                image_records.append({"id": image_id, "file_name": f"{split}.jpg", "width": 10, "height": 10})
            if split == "train" and augmented:
                image_records.append({
                    "id": image_id + 1,
                    "file_name": "train_aug01.jpg",
                    "width": 10,
                    "height": 10,
                    "source_image_id": image_id,
                    "augmentation_index": 1,
                })

            annotations = []
            for index, record in enumerate(image_records):
                Image.new("RGB", (10, 10)).save(root / "images" / split / record["file_name"])
                annotations.append({
                    "id": index + 1,
                    "image_id": record["id"],
                    "category_id": 1,
                    "bbox": [1, 1, 3, 3],
                    "area": 9,
                    "iscrowd": 0,
                })
                (root / "labels" / split / f"{Path(record['file_name']).stem}.txt").write_text(
                    "0 0.25000000 0.25000000 0.30000000 0.30000000\n",
                    encoding="utf-8",
                )
            coco = {
                "images": image_records,
                "annotations": annotations,
                "categories": categories,
            }
            (root / "annotations" / f"instances_{split}.json").write_text(
                json.dumps(coco), encoding="utf-8"
            )

        if augmented:
            config = {
                "name": "test-augmentation",
                "apply_to": ["train"],
                "variants_per_image": 1,
                "include_originals": include_originals,
            }
            (root / "augmentation_config.json").write_text(json.dumps(config), encoding="utf-8")

    def test_validates_coco_and_generated_augmentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._create_dataset(root, augmented=True)

            validate_coco_dataset(root, splits=("train", "val"))
            validate_yolo_dataset(root, expected_train_stems={"train"})

    def test_validates_unchanged_deduplicated_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._create_dataset(root)

            validate_yolo_dataset(root, expected_train_stems={"train"})

    def test_validates_an_individual_evaluation_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._create_dataset(root)

            validate_yolo_dataset(root, splits=("test",))

    def test_generated_dataset_matches_trainer_validation_contract(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            self._create_dataset(source)
            config = json.loads((repository / "configs" / "augmentation.json").read_text(encoding="utf-8"))
            config["variants_per_image"] = 1
            for transform in config["transformations"].values():
                transform["enabled"] = False
            config_path = root / "augmentation.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "augmented"

            create_augmented_dataset(source, output, config_path)
            validate_yolo_dataset(output, expected_train_stems={"train"})
            manifest_path = root / "train_manifest.json"
            manifest_path.write_text(json.dumps({"ordered_ids": ["train"]}), encoding="utf-8")
            from parasite_detection import yolo

            with patch.object(yolo, "TRAIN_MANIFEST", manifest_path):
                self.assertEqual(set(yolo._check_dataset(output)["names"]), set(range(8)))
            records = json.loads(
                (output / "annotations" / "instances_train.json").read_text(encoding="utf-8")
            )["images"]
            validate_loaded_image_paths(
                output, "train", [output / "images" / "train" / image["file_name"] for image in records]
            )
            with self.assertRaisesRegex(ValueError, "conservar todas las imágenes anotadas"):
                validate_loaded_image_paths(
                    output,
                    "train",
                    [output / "images" / "train" / image["file_name"] for image in records[:-1]],
                )
            for split in ("val", "test"):
                source_image = source / "images" / split / f"{split}.jpg"
                output_image = output / "images" / split / f"{split}.jpg"
                self.assertEqual(source_image.read_bytes(), output_image.read_bytes())

    def test_rejects_baseline_train_outside_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._create_dataset(root)

            with self.assertRaisesRegex(ValueError, "no coincide con el manifiesto"):
                validate_yolo_dataset(root, expected_train_stems={"another-image"})

    def test_supports_variants_without_original_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._create_dataset(root, augmented=True, include_originals=False)

            validate_yolo_dataset(root, expected_train_stems={"train"})

    def test_rejects_mismatched_coco_and_yolo_boxes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._create_dataset(root, augmented=True)
            label = root / "labels" / "train" / "train_aug01.txt"
            label.write_text("0 0.50000000 0.50000000 0.30000000 0.30000000\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cajas YOLO y COCO no coinciden"):
                validate_yolo_dataset(root, expected_train_stems={"train"})

    def test_rejects_incomplete_augmentation_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._create_dataset(root, augmented=True)
            config_path = root / "augmentation_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["variants_per_image"] = 2
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Faltan o se repiten variantes"):
                validate_yolo_dataset(root, expected_train_stems={"train"})


if __name__ == "__main__":
    unittest.main()
