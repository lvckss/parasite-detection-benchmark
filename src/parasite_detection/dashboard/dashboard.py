"""panel común con un modo vivo y un informe independiente del kernel"""

from __future__ import annotations

import base64
import copy
import csv
import html
import io
import json
import math
import os
import time
import uuid
from pathlib import Path

import anywidget
import traitlets
from IPython.display import display, update_display

from parasite_detection.paths import RESULTS

from .events import EventStore, TrainingEvent, clean


ASSETS = Path(__file__).parent
DEFAULT_VIEWS = [
    {"id": "performance", "title": "Evolución del rendimiento",
     "description": "Validación · mAP@50:95 y mAP@50",
     "series": [{"key": "map", "label": "mAP@50:95", "color": "#777cf4"},
                {"key": "map50", "label": "mAP@50", "color": "#25b4a4"}]},
    {"id": "loss", "title": "Evolución de pérdidas",
     "description": "Las pérdidas pertenecen a este experimento y a su definición de loss",
     "series": [{"key": "train_loss", "label": "Train loss", "color": "#777cf4"},
                {"key": "val_loss", "label": "Validation loss", "color": "#e9a15f"}]},
]


def _number(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"


def _clock(seconds):
    seconds = max(0, int(seconds or 0))
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class _DashboardWidget(anywidget.AnyWidget):
    _esm = ASSETS / "frontend.js"
    _css = ASSETS / "style.css"
    snapshot = traitlets.Dict().tag(sync=True)
    packet = traitlets.Dict().tag(sync=True)
    selected_view = traitlets.Int(0).tag(sync=True)

    def __init__(self, owner):
        self.owner = owner
        self.display_id = "training-dashboard-" + uuid.uuid4().hex
        self.shown = False
        self.ended = False
        # leemos ambos recursos juntos para no reutilizar css de una instancia anterior
        super().__init__(snapshot=clean(owner.state),
                         _esm=(ASSETS / "frontend.js").read_text(encoding="utf-8"),
                         _css=(ASSETS / "style.css").read_text(encoding="utf-8"))

    def _ipython_display_(self):
        # el id de display permite sustituir el widget por un output estático al terminar
        if self.owner.state["mode"] == "report":
            bundle = {"text/html": self.owner.snapshot_html(), "text/plain": self.owner.summary()}
            metadata = {}
        else:
            representation = self._repr_mimebundle_()
            bundle, metadata = representation if isinstance(representation, tuple) else (representation, {})
            bundle["text/html"] = self.owner.snapshot_html()
        if self.shown:
            update_display(bundle, raw=True, display_id=self.display_id)
        else:
            display(bundle, raw=True, metadata=metadata, display_id=self.display_id)
            self.shown = True

    def persist_output(self):
        if self.shown:
            update_display({"text/html": self.owner.snapshot_html(), "text/plain": self.owner.summary()},
                           raw=True, display_id=self.display_id)
        # el output final ya no referencia un modelo de widget que pueda desaparecer
        self.ended = True
        self.close()


class TrainingDashboard:
    def __init__(self, model_name, run_name, *, total_epochs=0, report_dir=None, views=None):
        self.state = {
            "version": 1, "model_name": str(model_name), "run_name": str(run_name), "mode": "live",
            "phase": "preparing", "phase_label": "Preparando", "epoch": 0, "total_epochs": total_epochs,
            "batch": 0, "batches": 0, "history": [], "best": {"map": None, "epoch": None},
            "elapsed_seconds": 0.0, "eta_seconds": None, "evaluation": {}, "metric_sources": {}, "config": {},
            "views": copy.deepcopy(views or DEFAULT_VIEWS), "logs": [], "checkpoint": None,
            "outcome": None, "error": None,
        }
        self.report_dir = Path(report_dir) if report_dir else None
        self.store = None
        self._widget = None
        self._started = time.monotonic()
        self._offset = 0.0
        self._last_progress = 0.0
        self._last_log = 0.0
        self._pending = []
        self._snapshot_cache = None
        self._eta_base_count = 0
        self._eta_base_time = 0.0

    @property
    def widget(self):
        if self._widget is None:
            self._widget = _DashboardWidget(self)
        return self._widget

    def summary(self):
        return (f"{self.state['model_name']} · {self.state['run_name']} · {self.state['phase']} · "
                f"época {self.state['epoch']}/{self.state['total_epochs']} · "
                f"mAP {_number(self.state['best']['map'])}")

    def bind(self, directory, *, resume=False):
        directory = Path(directory)
        if (directory / "dashboard_events.jsonl").exists() and not resume:
            raise FileExistsError(f"Ya existe un historial visual en {directory}")
        self.report_dir = directory
        self.store = EventStore(directory)
        for event in self._pending:
            self.store.append(event)
        self._pending.clear()

    def add_view(self, identifier, title, series, description=""):
        if any(v["id"] == identifier for v in self.state["views"]):
            raise ValueError(f"Ya existe la vista {identifier}")
        self.state["views"].append({"id": identifier, "title": title, "series": clean(series), "description": description})
        self._snapshot_cache = None
        self._publish("views", full=True)

    def emit(self, event, **data):
        if isinstance(event, str):
            event = TrainingEvent(event, data)
        payload = clean(event.data)
        self.state["elapsed_seconds"] = self._offset + time.monotonic() - self._started
        if event.kind == "start":
            self.state.update(payload)
            if "history" in payload:
                # al reanudar estimamos el tiempo restante con las épocas de esta sesión
                self._eta_base_count = len(payload["history"])
                self._eta_base_time = self.state["elapsed_seconds"]
        elif event.kind == "status":
            self.state.update(payload)
            self.state["batch"] = self.state["batches"] = 0
        elif event.kind == "progress":
            self.state.update(payload)
        elif event.kind == "epoch":
            row = payload["metrics"]
            epoch = int(row["epoch"])
            records = {int(r["epoch"]): r for r in self.state["history"]}
            records[epoch] = row
            self.state["history"] = [records[i] for i in sorted(records)]
            self.state["epoch"] = epoch
            value = row.get("map")
            previous = self.state["best"]["map"]
            if value is not None and (previous is None or value > previous):
                self.state["best"] = {"map": value, "epoch": epoch}
            if payload.get("best"):
                self.state["best"] = payload["best"]
            if payload.get("checkpoint"):
                self.state["checkpoint"] = payload["checkpoint"]
            done = len(records) - self._eta_base_count
            self.state["eta_seconds"] = ((self.state["elapsed_seconds"] - self._eta_base_time) / done
                                         * max(0, self.state["total_epochs"] - epoch)) if done > 0 else None
        elif event.kind == "log":
            self.state["logs"] = (self.state["logs"] + [str(payload["message"])])[-40:]
        elif event.kind in {"finish", "error"}:
            self.state.update(payload)
            self.state["phase"] = "error" if event.kind == "error" else "finished"
            self.state["mode"] = "report"
            self.state["eta_seconds"] = None
        else:
            raise ValueError(f"Tipo de evento no reconocido: {event.kind}")
        if event.kind == "error":
            # los errores permanecen en el notebook sin crear ni modificar archivos de reporte
            self._snapshot_cache = None
            self._pending.clear()
            self._publish("error", full=True)
            if self._widget:
                self._widget.persist_output()
            return
        if self.store:
            self.store.append(event)
        elif event.kind not in {"progress", "log"}:
            self._pending.append(event)
        full = event.kind in {"start", "epoch", "finish"}
        self._publish(event.kind, full=full)
        if event.kind in {"epoch", "finish"}:
            self._snapshot_cache = None
            if self.store:
                self.store.snapshot(self.state)
                self.export_html()
        if event.kind == "finish":
            if self.report_dir is None:
                self.report_dir = RESULTS / "reports" / (self.state["run_name"] + "-" + uuid.uuid4().hex[:8])
            if self.store is None:
                self.store = EventStore(self.report_dir)
                for pending in self._pending:
                    self.store.append(pending)
                self._pending.clear()
                self.store.snapshot(self.state)
                self.export_html()
            if self._widget:
                self._widget.persist_output()

    def _publish(self, kind, full=False):
        if self._widget and not self._widget.ended:
            state = self.state
            if kind == "progress":
                # los batches solo envían los campos pequeños que realmente cambian
                keys = ("batch", "batches", "epoch", "phase", "phase_label", "elapsed_seconds", "eta_seconds")
                state = {k: state[k] for k in keys}
            elif kind == "log":
                state = {"logs": state["logs"]}
            elif not full:
                state = {k: v for k, v in state.items() if k not in {"history", "views", "best"}}
            state = clean(state)
            self._widget.packet = {"kind": kind, "state": state, "sequence": time.monotonic_ns()}
            if full or not self._widget.shown:
                self._widget.snapshot = clean(self.state)

    def progress(self, batch, batches, **data):
        now = time.monotonic()
        if batch < batches and now - self._last_progress < 0.25:
            return
        self._last_progress = now
        self.emit("progress", batch=batch, batches=batches, **data)

    def log(self, message):
        # guardamos un anillo pequeño y limitamos las notificaciones del canal secundario
        self.state["logs"] = (self.state["logs"] + [str(message)[:500]])[-40:]
        if time.monotonic() - self._last_log > 1:
            self._last_log = time.monotonic()
            self._publish("log")

    def export_html(self, path=None):
        if path is None and self.report_dir is None:
            raise ValueError("Indica una ruta o un report_dir para guardar el informe")
        path = Path(path) if path else self.report_dir / "report.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        state = clean(self.state)
        state["mode"] = "report"
        if state["phase"] in {"preparing", "training", "validating"}:
            state["phase"] = "unknown"
            state["phase_label"] = "Última validación guardada"
        serialized = json.dumps(state, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        script = (ASSETS / "frontend.js").read_text(encoding="utf-8")
        css = (ASSETS / "style.css").read_text(encoding="utf-8")
        # el html incluye todos sus datos y recursos y no importa módulos desde la red
        document = ('<!doctype html><html lang="es"><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width,initial-scale=1">'
                    f"<title>{html.escape(self.state['model_name'])} · {html.escape(self.state['run_name'])}</title>"
                    f"<style>body{{margin:0;padding:24px;background:#f5f5f6}}@media(prefers-color-scheme:dark){{body{{background:#0c0c0e}}}}{css}</style>"
                    f'<main id="dashboard"><noscript>{self.snapshot_html(include_interactive=False)}</noscript></main>'
                    f'<script id="training-data" type="application/json">{serialized}</script>'
                    f'<script type="module">{script}\nwindow.trainingReport = mount(document.getElementById("dashboard"),JSON.parse(document.getElementById("training-data").textContent),{{pageTheme:true}});</script></html>')
        temporary = path.with_suffix(".html.tmp")
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(path)
        return path

    def _chart_png(self, view):
        # solo generamos imágenes al consolidar el output y nunca durante los batches
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.ticker import MaxNLocator
        figure = Figure(figsize=(9, 2.7), dpi=110, facecolor="none")
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        ax.set_facecolor("none")
        history = self.state["history"]
        for series in view["series"]:
            ax.plot([r["epoch"] for r in history],
                    [r.get(series["key"]) if r.get(series["key"]) is not None else float("nan") for r in history],
                    color=series["color"], linewidth=2, label=series["label"])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.tick_params(colors="#81818c", labelsize=8, length=0, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color="#81818c", alpha=0.16)
        ax.set_xlabel("Época", color="#81818c", fontsize=8)
        ax.legend(frameon=False, fontsize=8, labelcolor="#81818c", loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
        if not history:
            ax.text(.5, .5, "Sin validaciones registradas", transform=ax.transAxes, ha="center", color="#81818c", fontsize=10)
        figure.tight_layout(pad=1.5)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", transparent=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def snapshot_html(self, include_interactive=True):
        if self._snapshot_cache is None:
            self._snapshot_cache = [self._chart_png(view) for view in self.state["views"]]
        state = self.state
        esc = lambda value: html.escape(str(value), quote=True)
        labels = {"preparing": "Preparando", "training": "Entrenando", "validating": "Validando",
                  "finished": "Early stopped" if state["outcome"] == "early_stopped" else "Completed",
                  "error": "Error", "unknown": "Historial guardado"}
        row = state["history"][-1] if state["history"] else {}
        values = [(key, label, state["best"]["map"] if key == "best" else row.get(key))
                  for key, label in [("map", "mAP@50:95"), ("best", "Mejor mAP@50:95"),
                                     ("map50", "mAP@50"), ("precision", "Precision"), ("recall", "Recall")]]
        cards = "".join(f'<div class="td-card"><div class="td-label">{label}</div><div class="td-value">{_number(value)}</div>'
                        f'<div class="td-sub">{esc("Época " + str(state["best"]["epoch"]) if key == "best" and state["best"]["epoch"] else state["metric_sources"].get(key, "No registrado" if value is None else "Última validación"))}</div></div>'
                        for key, label, value in values)
        identifier = "snapshot-" + uuid.uuid4().hex
        count = len(state["views"])
        selected = min(self._widget.selected_view if self._widget else 0, count - 1)
        radios = "".join(f'<input class="td-radio" aria-label="{esc(view["title"])}" type="radio" name="{identifier}" id="{identifier}-{i}" {"checked" if i == selected else ""}>'
                         for i, view in enumerate(state["views"]))
        rules = "".join(f'#{identifier}-{i}:checked~.td-snapshot .td-snapshot-track{{transform:translateX(-{i * 100}%)}}' for i in range(count))
        charts = "".join(f'<section><div class="td-chart-head"><div class="td-chart-title">{esc(view["title"])}</div>'
                         f'<span class="td-nav">{i + 1}/{count}</span></div>'
                         f'<img class="td-static-chart" alt="{esc(view["title"])}" src="data:image/png;base64,{image}"></section>'
                         for i, (view, image) in enumerate(zip(state["views"], self._snapshot_cache)))
        controls = "".join(f'<label for="{identifier}-{i}">{esc(view["title"])}</label>' for i, view in enumerate(state["views"]))
        error = f'<div class="td-error">{esc(state["error"])}</div>' if state["error"] else ""
        checkpoint = f'<div class="td-checkpoint">Mejor checkpoint · {esc(state["checkpoint"])}</div>' if state["checkpoint"] else ""
        footer = ""
        if include_interactive and self.report_dir and (self.report_dir / "report.html").is_file():
            link = os.path.relpath(self.report_dir / "report.html", Path.cwd()).replace(os.sep, "/")
            from urllib.parse import quote
            footer = f'<a href="{quote(link)}" target="_blank" rel="noopener">Abrir informe interactivo ↗</a>'
        markup = (f'<style>{(ASSETS / "style.css").read_text(encoding="utf-8")}{rules}</style>'
                f'<article class="td" data-phase="{esc(state["phase"])}"><header class="td-top"><div>'
                f'<h2>{esc(state["model_name"])}</h2>'
                f'<div class="td-run">{esc(state["run_name"])}</div></div><div class="td-status">{labels[state["phase"]]}</div></header>'
                f'<div class="td-progress"><div class="td-progress-copy"><strong>Época {state["epoch"]} / {state["total_epochs"] or "—"}</strong>'
                f'<span>{len(state["history"])} validaciones guardadas</span></div><div class="td-bar"><div class="td-bar-fill" style="width:{100 if state["phase"] == "finished" else 0}%"></div></div></div>'
                f'{error}<section class="td-cards">{cards}</section>{radios}<div class="td-snapshot"><div class="td-snapshot-track">{charts}</div></div>'
                f'<nav class="td-static-controls" aria-label="Gráficas guardadas">{controls}</nav>{checkpoint}'
                f'<footer class="td-footer"><span>Transcurrido {_clock(state["elapsed_seconds"])}</span>{footer}<span>{esc(state["evaluation"].get("label", "Métricas guardadas"))}</span></footer>'
                f'<details class="td-details"><summary>Detalles del experimento y mensajes técnicos</summary><pre>{esc(json.dumps({"evaluation": state["evaluation"], "config": state["config"]}, ensure_ascii=False, indent=2))}\n{esc(chr(10).join(state["logs"]))}</pre></details></article>')
        if include_interactive and state["mode"] == "report":
            # algunos visores permiten javascript en html confiable y otros solo muestran el fallback
            # la copia estática permanece hasta que el frontend se haya construido correctamente
            data = json.dumps(clean(state), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
            script = (ASSETS / "frontend.js").read_text(encoding="utf-8")
            style_end = markup.index("</style>") + len("</style>")
            return (markup[:style_end] + f'<div id="{identifier}">{markup[style_end:]}</div><script type="module">{script}\n'
                    f'const savedHost=document.getElementById("{identifier}");'
                    f'if(savedHost){{const enhanced=document.createElement("div");mount(enhanced,{data});'
                    'savedHost.replaceChildren(enhanced);}</script>')
        return markup

    @classmethod
    def from_run(cls, directory, *, model_name=None, outcome=None):
        directory = Path(directory)
        settings_path = directory / "settings.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
        snapshot = directory / "dashboard_state.json"
        saved = json.loads(snapshot.read_text(encoding="utf-8")) if snapshot.exists() else {}
        dashboard = cls(model_name or saved.get("model_name", "Experimento"),
                        saved.get("run_name", directory.name), total_epochs=settings.get("epochs", saved.get("total_epochs", 0)))
        dashboard.state.update(saved)
        dashboard.state["config"] = settings
        dashboard.state["total_epochs"] = settings.get("epochs", dashboard.state["total_epochs"])
        if model_name:
            dashboard.state["model_name"] = model_name
        dashboard.state["mode"] = "report"
        # el historial se reconstruye desde el csv fuente y no desde las curvas del widget
        with (directory / "training.csv").open(newline="", encoding="utf-8") as file:
            history = []
            for row in csv.DictReader(file):
                values = {}
                for key, value in row.items():
                    try:
                        values[key] = int(value) if key == "epoch" else float(value) if value else None
                    except (ValueError, TypeError):
                        values[key] = value
                history.append(clean(values))
        by_epoch = {r["epoch"]: r for r in history}
        # solo recuperamos extras visuales que no existen en el csv fuente
        for old in saved.get("history", []):
            row = by_epoch.get(old.get("epoch"))
            if row and row.get("map") == old.get("map"):
                for key in ("precision", "recall"):
                    if key not in row and key in old:
                        row[key] = old[key]
        dashboard.state["history"] = [by_epoch[i] for i in sorted(by_epoch)]
        dashboard.state["epoch"] = max(by_epoch, default=0)
        if saved.get("phase") == "error":
            dashboard.state["epoch"] = saved.get("epoch", dashboard.state["epoch"])
        scored = [r for r in history if r.get("map") is not None]
        if scored:
            best = max(scored, key=lambda r: r["map"])
            previous = saved.get("best", {})
            tied = by_epoch.get(previous.get("epoch"))
            if tied and tied.get("map") == best["map"]:
                best = tied
            dashboard.state["best"] = {"map": best["map"], "epoch": best["epoch"]}
        if outcome:
            dashboard.state.update(phase="finished", outcome=outcome)
        elif not saved or saved.get("phase") not in {"finished", "error"}:
            # un csv antiguo por sí solo no demuestra que la ejecución terminase correctamente
            dashboard.state.update(phase="unknown", outcome=None)
        dashboard.report_dir = directory
        return dashboard
