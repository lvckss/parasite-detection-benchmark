"""pruebas del contrato, persistencia e integración sin entrenar detectores completos"""

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from parasite_detection.dashboard import TrainingDashboard
from parasite_detection.dashboard.adapters import TrainingSession, UltralyticsEvents
from parasite_detection.dashboard.demo import simulate_training
from parasite_detection.dashboard.events import EventStore, TrainingEvent, clean
from parasite_detection import faster_rcnn as rcnn


class DashboardTests(unittest.TestCase):
    def test_report_restore_and_csv_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "demo"
            dashboard = TrainingDashboard("Modelo <demo>", "baseline")
            history = simulate_training(dashboard, directory, epochs=4, early_stop=3, delay=0)
            restored = TrainingDashboard.from_run(directory)
            self.assertEqual(restored.state["history"], history)
            self.assertEqual(restored.state["phase"], "finished")
            self.assertEqual(restored.state["outcome"], "early_stopped")
            self.assertEqual(restored.state["best"]["epoch"], 3)
            report = (directory / "report.html").read_text()
            self.assertIn('type="module"', report)
            self.assertNotIn('<script src=', report)
            self.assertNotIn('cdn.', report)
            self.assertIn("data:image/png;base64,", report)
            self.assertIn("&lt;demo&gt;", restored.snapshot_html())
            # una copia visual obsoleta nunca sustituye a los valores del csv
            state = json.loads((directory / "dashboard_state.json").read_text())
            state["history"][-1]["map"] = 999
            (directory / "dashboard_state.json").write_text(json.dumps(state))
            self.assertEqual(TrainingDashboard.from_run(directory).state["history"][-1]["map"], history[-1]["map"])

    def test_failure_keeps_previous_validations(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "failed"
            dashboard = TrainingDashboard("Demo", "failed")
            with self.assertRaisesRegex(RuntimeError, "Error simulado"):
                simulate_training(dashboard, directory, epochs=4, delay=0, fail_at=3)
            restored = TrainingDashboard.from_run(directory)
            self.assertEqual(dashboard.state["phase"], "error")
            self.assertIn("Error simulado", dashboard.state["error"])
            self.assertEqual(restored.state["phase"], "unknown")
            self.assertEqual(len(restored.state["history"]), 2)
            self.assertNotIn("Error simulado", (directory / "report.html").read_text())
            self.assertFalse(any(event.kind == "error" for event in dashboard.store.read()))

    def test_error_does_not_create_reports_or_change_saved_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for report_dir in (None, root / "unused"):
                dashboard = TrainingDashboard("Demo", "failed", report_dir=report_dir)
                with patch("parasite_detection.dashboard.dashboard.RESULTS", root):
                    dashboard.emit("error", error="fallo antes de entrenar", outcome="error")
                self.assertEqual(dashboard.state["phase"], "error")
                self.assertEqual(list(root.iterdir()), [])
            dashboard = TrainingDashboard("Demo", "existing")
            dashboard.bind(root / "existing")
            dashboard.emit("epoch", metrics={"epoch": 1, "map": .5})
            before = {p.name: p.read_bytes() for p in dashboard.report_dir.iterdir()}
            dashboard.emit("error", error="fallo después de validar", outcome="error")
            after = {p.name: p.read_bytes() for p in dashboard.report_dir.iterdir()}
            self.assertEqual(before, after)

    def test_progress_is_throttled_and_not_written_per_batch(self):
        dashboard = TrainingDashboard("Demo", "throttle")
        with tempfile.TemporaryDirectory() as temporary:
            dashboard.bind(temporary)
            with patch("parasite_detection.dashboard.dashboard.time.monotonic", return_value=100):
                for batch in range(1, 101):
                    dashboard.progress(batch, 100)
            self.assertEqual(dashboard.state["batch"], 100)
            self.assertFalse(dashboard.store.path.exists())

    def test_store_tolerates_interrupted_line_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = EventStore(temporary)
            store.append(TrainingEvent("epoch", {"metrics": {"epoch": 1, "map": float("nan")}}))
            with store.path.open("a") as file:
                file.write('{"kind":')
            self.assertIsNone(store.read()[0].data["metrics"]["map"])
            self.assertIsNone(clean(float("inf")))

    def test_ui_failure_does_not_escape_and_capture_restores_streams(self):
        dashboard = TrainingDashboard("Demo", "failure")
        before = __import__("sys").stdout
        with contextlib.redirect_stderr(io.StringIO()):
            with TrainingSession(dashboard, SimpleNamespace(epochs=1)) as session:
                with patch.object(dashboard, "progress", side_effect=RuntimeError("render failed")):
                    session.progress(1, 2)
                print("mensaje técnico")
        self.assertIs(__import__("sys").stdout, before)
        self.assertEqual(len(session.warnings), 1)
        self.assertIn("mensaje técnico", dashboard.state["logs"])

    def test_ultralytics_adapter_observes_csv_and_skips_final_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            native_path = directory / "native.csv"
            native_path.write_text("epoch,metrics/precision(B),metrics/recall(B)\n1,0.9,0.8\n2,0.7,0.6\n")
            (directory / "training.csv").write_text("epoch,map,map50,train_loss,val_loss,lr\n1,0.5,0.6,2,3,0.01\n2,0.5,0.6,1,2,0.005\n")
            dashboard = TrainingDashboard("YOLO", "toy")
            with TrainingSession(dashboard, SimpleNamespace(epochs=2)) as session:
                adapter = UltralyticsEvents(session, directory, resume=False, evaluation={"label": "COCOeval"})
                trainer = SimpleNamespace(csv=native_path, validator=SimpleNamespace(training=True), best=directory / "best.pt")
                adapter.on_fit_epoch_end(trainer)
                self.assertEqual(dashboard.state["best"]["epoch"], 2)
                self.assertEqual(dashboard.state["history"][-1]["precision"], .7)
                trainer.validator.training = False
                adapter.on_fit_epoch_end(trainer)
                self.assertEqual(len(dashboard.state["history"]), 1)


class _TinyDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return torch.full((3, 8, 8), (index + 1) / 5), {"image_id": torch.tensor(index + 1)}


class _TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.75))
        self.roi_heads = SimpleNamespace(nms_thresh=.5, score_thresh=.05, detections_per_img=100)

    def forward(self, images, targets=None):
        return {"loss": sum((self.weight * image.mean() - .2).square() for image in images)}


class TrainerParityTests(unittest.TestCase):
    def test_rcnn_observer_does_not_change_history_or_checkpoint(self):
        loader = DataLoader(_TinyDataset(), batch_size=1, collate_fn=rcnn.collate_batch)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = rcnn.TrainConfig(data_dir=root, epochs=2, warmup_batches=1, run_name="toy", device="cpu")
            records = []
            for name in ("terminal", "dashboard"):
                run = root / name
                observer = TrainingDashboard("Tiny detector", "toy") if name == "dashboard" else None
                with patch.object(rcnn, "MODELS", run / "models"), patch.object(rcnn, "RESULTS", run / "results"), \
                     patch.object(rcnn, "build_model", side_effect=lambda *a, **k: _TinyDetector()), \
                     patch.object(rcnn, "create_dataloaders", return_value=(loader, loader)), \
                     patch.object(rcnn, "evaluate_model", return_value={"map": .5, "map50": .7}), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    model, history = rcnn.train_model(config, dashboard=observer)
                checkpoint = torch.load(run / "models/trained/fasterrcnn_v2/toy/last_checkpoint.pth", weights_only=True)
                records.append((model.weight.detach().clone(), history, checkpoint))
            torch.testing.assert_close(records[0][0], records[1][0], rtol=0, atol=0)
            self.assertEqual(records[0][1], records[1][1])
            self.assertEqual(records[0][2]["scheduler"], records[1][2]["scheduler"])
            self.assertEqual(records[0][2]["best_epoch"], records[1][2]["best_epoch"])
            torch.testing.assert_close(records[0][2]["optimizer"]["state"][0]["momentum_buffer"],
                                       records[1][2]["optimizer"]["state"][0]["momentum_buffer"], rtol=0, atol=0)


@unittest.skipUnless(os.environ.get("RUN_DASHBOARD_NOTEBOOK_CHECK") == "1", "requiere iniciar un kernel local")
class NotebookPersistenceTests(unittest.TestCase):
    def test_saved_output_survives_kernel_shutdown(self):
        import nbformat
        from nbclient import NotebookClient

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
                "from IPython.display import display\n"
                "from parasite_detection.dashboard import TrainingDashboard\n"
                "from parasite_detection.dashboard.demo import simulate_training\n"
                "dashboard = TrainingDashboard('Demo', 'kernel-off')\n"
                "display(dashboard.widget)\n"
                f"history = simulate_training(dashboard, {str(directory / 'run')!r}, epochs=3, delay=0)"
            )])
            client = NotebookClient(notebook, kernel_name="python3", timeout=90)
            client.execute()
            path = directory / "saved.ipynb"
            nbformat.write(notebook, path)
            self.assertIsNone(client.km)
            saved = nbformat.read(path, as_version=4)
            outputs = [o for o in saved.cells[0].outputs if o.output_type == "display_data"]
            self.assertEqual(len(outputs), 1)
            self.assertNotIn("application/vnd.jupyter.widget-view+json", outputs[0].data)
            self.assertIn("Completed", outputs[0].data["text/html"])
            self.assertIn("data:image/png;base64,", outputs[0].data["text/html"])
            self.assertEqual(len(TrainingDashboard.from_run(directory / "run").state["history"]), 3)


if __name__ == "__main__":
    unittest.main()
