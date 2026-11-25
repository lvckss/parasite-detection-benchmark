"""simulación determinista para revisar el panel sin datos ni entrenamiento"""

import csv
import json
import math
import time
from pathlib import Path

from .dashboard import TrainingDashboard


def simulate_training(dashboard, directory, *, epochs=12, delay=0.04, early_stop=None, fail_at=None):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"La demo necesita una carpeta vacía: {directory}")
    if epochs < 1 or delay < 0:
        raise ValueError("epochs debe ser positivo y delay no puede ser negativo")
    dashboard.bind(directory)
    settings = {"epochs": epochs, "simulation": True, "run_name": dashboard.state["run_name"]}
    (directory / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    dashboard.emit("start", total_epochs=epochs, evaluation={
        "label": "SIMULACIÓN · sin modelo ni evaluación real", "simulation": True,
        "loss_reporting": "valores sintéticos para comprobar la visualización",
    }, metric_sources={key: "Simulado" for key in ("map", "map50", "precision", "recall")})
    history = []
    try:
        for epoch in range(1, min(epochs, early_stop or epochs) + 1):
            dashboard.emit("status", phase="training", phase_label="Entrenando · simulación", epoch=epoch)
            for batch in range(1, 9):
                dashboard.progress(batch, 8)
                time.sleep(delay)
            dashboard.emit("status", phase="validating", phase_label="Validando · simulación")
            for batch in range(1, 4):
                dashboard.progress(batch, 3)
                time.sleep(delay)
            if epoch == fail_at:
                raise RuntimeError("Error simulado para comprobar la conservación del historial")
            row = {
                "epoch": epoch, "map": .68 * (1 - math.exp(-epoch / 3.3)),
                "map50": .86 * (1 - math.exp(-epoch / 2.7)),
                "precision": .82 * (1 - math.exp(-epoch / 2)),
                "recall": .78 * (1 - math.exp(-epoch / 3)),
                "train_loss": 4.2 * math.exp(-epoch / 4) + .3,
                "val_loss": 4.5 * math.exp(-epoch / 4.4) + .4,
                "lr": .0025 * (1 + math.cos(math.pi * epoch / epochs)) / 2,
            }
            history.append(row)
            # igual que los adaptadores reales publicamos después de consolidar el csv
            with (directory / "training.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            dashboard.emit("epoch", metrics=row)
        dashboard.emit("finish", checkpoint="No existe checkpoint · simulación",
                       outcome="early_stopped" if len(history) < epochs else "completed")
    except BaseException as error:
        dashboard.emit("error", error=f"{type(error).__name__}: {error}", outcome="error")
        raise
    return history


def create_demo(directory, *, delay=0.04, epochs=12):
    dashboard = TrainingDashboard("Detector · demo", "simulated-baseline", total_epochs=epochs)
    simulate_training(dashboard, directory, delay=delay, epochs=epochs)
    return dashboard
