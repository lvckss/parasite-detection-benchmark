"""control optativo con el entrenador nativo en cpu y archivos temporales"""

import contextlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import SubsetRandomSampler

from parasite_detection.dashboard import TrainingDashboard


@unittest.skipUnless(os.environ.get("RUN_YOLO_DASHBOARD_CHECK") == "1", "control nativo optativo")
class YoloDashboardIntegrationTests(unittest.TestCase):
    def test_native_weights_and_metrics_are_identical(self):
        from parasite_detection import yolo

        if not yolo.COCO_WEIGHTS.is_file() or not yolo.DEDUP_DATA.is_dir():
            self.skipTest("requiere los pesos y el dataset depurado locales")
        original_loader = yolo.ParasiteTrainer.get_dataloader

        def tiny_loader(trainer, dataset_path, batch_size=16, rank=0, mode="train"):
            loader = original_loader(trainer, dataset_path, batch_size, rank, mode)
            if mode == "train":
                # validamos las 1157 imágenes y solo acortamos el muestreo de esta prueba
                loader.batch_sampler.sampler.sampler = SubsetRandomSampler([0, 1, 2, 3])
                loader.reset()
            return loader

        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data = root / "data"
                (data / "annotations").mkdir(parents=True)
                shutil.copy2(yolo.DEDUP_DATA / "class_mapping.json", data / "class_mapping.json")
                for split in ("train", "val"):
                    (data / "images" / split).mkdir(parents=True)
                    shutil.copytree(yolo.DEDUP_DATA / "labels" / split, data / "labels" / split)
                    shutil.copy2(yolo.DEDUP_DATA / "annotations" / f"instances_{split}.json",
                                 data / "annotations" / f"instances_{split}.json")
                    for image in (yolo.DEDUP_DATA / "images" / split).iterdir():
                        if image.is_file():
                            (data / "images" / split / image.name).symlink_to(image.resolve())
                records = []
                for name in ("terminal", "dashboard"):
                    run = root / name
                    observer = TrainingDashboard("YOLOv10-S", "parity") if name == "dashboard" else None
                    config = yolo.TrainConfig(data_dir=data, run_name="parity", epochs=2, batch_size=2,
                                              image_size=64, device="cpu", num_workers=0, amp=False, patience=0)
                    with patch.object(yolo, "MODELS", run / "models"), patch.object(yolo, "RESULTS", run / "results"), \
                         patch.object(yolo.ParasiteTrainer, "get_dataloader", tiny_loader), \
                         (root / f"{name}.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        model, history = yolo.train_model(config, dashboard=observer)
                    checkpoint = torch.load(model.trainer.best, map_location="cpu", weights_only=False)
                    records.append((history, checkpoint["model"].state_dict()))
                    if observer:
                        self.assertEqual(observer.state["phase"], "finished")
                        self.assertEqual(len(observer.state["history"]), 2)
                        self.assertEqual(observer.state["epoch"], 2)
                        self.assertTrue(observer.state["metric_sources"]["precision"].startswith("Ultralytics"))
                        self.assertIsNotNone(observer.state["history"][-1]["precision"])
                        self.assertTrue((observer.report_dir / "report.html").is_file())
                self.assertEqual(records[0][0], records[1][0])
                self.assertEqual(records[0][1].keys(), records[1][1].keys())
                for key in records[0][1]:
                    torch.testing.assert_close(records[0][1][key], records[1][1][key], rtol=0, atol=0, equal_nan=True)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
