"""contrato de eventos y almacenamiento independiente de los entrenadores"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingEvent:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self):
        return {"version": 1, "kind": self.kind, "data": clean(self.data), "timestamp": self.timestamp}


def clean(value):
    # json no debe contener nan ni infinitos y las métricas ausentes siguen siendo ausentes
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return clean(value.item())
    return str(value)


class EventStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "dashboard_events.jsonl"

    def append(self, event):
        # el progreso fino se queda en memoria para no escribir por cada batch
        if event.kind in {"progress", "log"}:
            return
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")

    def snapshot(self, state):
        path = self.directory / "dashboard_state.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(clean(state), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def read(self):
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                events.append(TrainingEvent(record["kind"], record["data"], record["timestamp"]))
            except (ValueError, KeyError):
                # una última línea incompleta tras una interrupción no borra el historial anterior
                continue
        return events
