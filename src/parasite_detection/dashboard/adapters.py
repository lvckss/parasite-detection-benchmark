"""observadores pequeños que traducen los entrenadores al contrato del panel"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import sys
import time
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path


def read_history(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return [{k.strip(): int(v) if k.strip() == "epoch" else float(v)
                 for k, v in row.items()} for row in csv.DictReader(file)]


class _Messages(io.TextIOBase):
    def __init__(self, session):
        self.session = session
        self.pending = ""

    def write(self, text):
        # las barras de terminal usan retornos de carro y secuencias ansi
        plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(text)).replace("\r", "\n")
        pieces = (self.pending + plain).split("\n")
        self.pending = pieces.pop()[-1000:]
        for line in pieces:
            if line.strip():
                self.session.call("log", line.strip())
        return len(text)

    def flush(self):
        if self.pending.strip():
            self.session.call("log", self.pending.strip())
        self.pending = ""


class TrainingSession:
    """captura la salida y evita que un fallo visual interrumpa el entrenamiento"""

    def __init__(self, dashboard, config):
        self.dashboard = dashboard
        self.config = config
        self.warnings = []
        self.stack = ExitStack()
        self.messages = _Messages(self)

    def call(self, method, *args, **kwargs):
        try:
            return getattr(self.dashboard, method)(*args, **kwargs)
        except Exception as error:
            # el csv y los checkpoints siguen escribiéndose aunque falle la interfaz
            message = f"Dashboard · {method} · {type(error).__name__} · {error}"
            if message not in self.warnings:
                self.warnings.append(message)

    def __enter__(self):
        self.dashboard._started = time.monotonic()
        self.call("emit", "start", total_epochs=self.config.epochs)
        self.stack.enter_context(redirect_stdout(self.messages))
        self.stack.enter_context(redirect_stderr(self.messages))
        # los handlers ya creados conservan la referencia al stream original
        self.handlers = []
        logger = logging.getLogger("ultralytics")
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                self.handlers.append((handler, handler.stream))
                handler.setStream(self.messages)
        self.native_utils = sys.modules.get("ultralytics.utils")
        if self.native_utils:
            self.verbose = self.native_utils.VERBOSE
            self.native_utils.VERBOSE = False
        return self

    def __exit__(self, error_type, error, traceback):
        self.messages.flush()
        if error is not None:
            self.call("emit", "error", error=f"{error_type.__name__}: {error}", outcome="error")
        for handler, stream in self.handlers:
            handler.setStream(stream)
        if self.native_utils:
            self.native_utils.VERBOSE = self.verbose
        self.stack.close()
        # los fallos de la interfaz se notifican una sola vez fuera de la captura
        for message in self.warnings:
            print(message, file=sys.stderr)
        return False

    def bind(self, directory, *, resume, evaluation, metric_sources):
        saved = Path(directory) / "dashboard_state.json"
        if resume and saved.exists():
            try:
                self.dashboard._offset = float(json.loads(saved.read_text(encoding="utf-8")).get("elapsed_seconds", 0))
            except (ValueError, OSError):
                pass
        self.call("bind", directory, resume=resume)
        settings = Path(directory) / "settings.json"
        config = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
        self.call("emit", "start", evaluation=evaluation, metric_sources=metric_sources, config=config)

    def restore(self, history, best=None):
        if history:
            self.call("emit", "start", history=history, epoch=int(history[-1]["epoch"]),
                      best=best or max(({"map": r["map"], "epoch": r["epoch"]} for r in history),
                                       key=lambda r: r["map"]))

    def status(self, phase, label, **data):
        self.call("emit", "status", phase=phase, phase_label=label, **data)

    def progress(self, batch, batches):
        self.call("progress", batch, batches)

    def epoch(self, metrics, best=None, checkpoint=None):
        self.call("emit", "epoch", metrics=metrics, best=best, checkpoint=checkpoint)

    def finish(self, history, checkpoint):
        self.messages.flush()
        actual = int(history[-1]["epoch"]) if history else 0
        self.call("emit", "finish", epoch=actual, checkpoint=str(checkpoint),
                  outcome="early_stopped" if actual < self.config.epochs else "completed")


class UltralyticsEvents:
    """callbacks sin imports de ultralytics ni acceso a gradientes o predicciones"""

    def __init__(self, session, directory, *, resume, evaluation):
        self.session = session
        self.directory = Path(directory)
        self.resume = resume
        self.evaluation = evaluation
        self.batch = 0

    def install(self, model):
        for name in ("on_train_start", "on_train_epoch_start", "on_train_batch_end",
                     "on_val_start", "on_val_batch_end", "on_fit_epoch_end"):
            model.add_callback(name, self._safe_callback(name))

    def _safe_callback(self, name):
        def observe(trainer):
            try:
                getattr(self, name)(trainer)
            except Exception as error:
                message = f"Dashboard · {name} · {type(error).__name__} · {error}"
                if message not in self.session.warnings:
                    self.session.warnings.append(message)
        return observe

    def on_train_start(self, trainer):
        self.session.bind(self.directory, resume=self.resume, evaluation=self.evaluation,
                          metric_sources={"map": "COCOeval · val", "map50": "COCOeval · val",
                                          "precision": "Ultralytics · val", "recall": "Ultralytics · val"})
        path = self.directory / "training.csv"
        if self.resume and path.exists():
            rows = [r for r in read_history(path) if r["epoch"] <= trainer.start_epoch]
            self.session.restore(rows)

    def on_train_epoch_start(self, trainer):
        self.batch = 0
        self.session.status("training", "Entrenando", epoch=trainer.epoch + 1, total_epochs=trainer.epochs)

    def on_train_batch_end(self, trainer):
        self.batch += 1
        self.session.progress(self.batch, len(trainer.train_loader))

    def on_val_start(self, validator):
        self.session.status("validating", "Validación COCO" if validator.training else "Validando el mejor checkpoint")

    def on_val_batch_end(self, validator):
        self.session.progress(validator.batch_i + 1, len(validator.dataloader))

    def on_fit_epoch_end(self, trainer):
        if not trainer.validator.training:
            return
        # este callback se instala después del que consolida el csv del proyecto
        rows = read_history(self.directory / "training.csv")
        row = rows[-1].copy()
        with Path(trainer.csv).open(newline="", encoding="utf-8") as file:
            native = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(file)]
        latest = next(r for r in reversed(native) if int(r["epoch"]) == row["epoch"])
        for key, source in (("precision", "metrics/precision(B)"), ("recall", "metrics/recall(B)")):
            if source in latest:
                row[key] = float(latest[source])
        # ultralytics guarda best también en los empates de fitness
        best = max(rows, key=lambda r: (r["map"], r["epoch"]))
        self.session.epoch(row, {"map": best["map"], "epoch": best["epoch"]}, str(trainer.best))
