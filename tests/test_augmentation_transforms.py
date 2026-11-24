import json
import unittest
from pathlib import Path

import numpy as np

from scripts.create_augmented_dataset import augment_image


class AugmentationTransformTests(unittest.TestCase):
    def test_records_selected_transforms_deterministically(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/augmentation.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for transform in config["transformations"].values():
            transform["enabled"] = False
        expected = [
            "horizontal_flip",
            "vertical_flip",
            "rotation",
            "scale",
            "contrast",
            "brightness",
        ]
        for name in expected:
            config["transformations"][name]["enabled"] = True
            config["transformations"][name]["probability"] = 1.0

        image = np.full((64, 64, 3), 127, dtype=np.uint8)
        boxes = [{"category_id": 1, "iscrowd": 0, "bbox": [20, 20, 24, 24]}]
        first_image, first_boxes, first_transforms = augment_image(image, boxes, config, seed=7)
        second_image, second_boxes, second_transforms = augment_image(image, boxes, config, seed=7)

        self.assertEqual(first_transforms, expected)
        self.assertEqual(second_transforms, expected)
        np.testing.assert_array_equal(first_image, second_image)
        self.assertEqual(first_boxes, second_boxes)


if __name__ == "__main__":
    unittest.main()
