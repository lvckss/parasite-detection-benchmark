"""Superpone las curvas de entrenamiento de varias ejecuciones en una misma figura.

Cada ejecución se indica como NOMBRE=RAÍZ, donde RAÍZ contiene
<modelo>/<NOMBRE>/training.csv (por ejemplo results/metrics). Genera dos figuras:
métricas de validación (mAP@50:95 y mAP@50) y pérdidas (train y val), con una fila
por modelo. Las pérdidas no se comparan entre modelos: cada fila tiene su propio eje.

    python scripts/evaluation/plot_training_comparison.py \
        --run baseline=results/metrics \
        --run aug_v1=results/parasites-runs-augmentation-001/metrics
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

MODELS = {"fasterrcnn_v2": "Faster R-CNN v2", "yolov10s": "YOLOv10-S"}
# orden fijo de la paleta categórica: el color sigue a la ejecución, no a su posición
COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600"]
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"


def read_history(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    def column(key):
        return [float(r[key]) if r.get(key) not in (None, "") else None for r in rows]
    return {"epoch": [int(r["epoch"]) for r in rows], **{k: column(k) for k in ("map", "map50", "train_loss", "val_loss")}}


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlabel("Época", color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bbbbbb")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_series(ax, history, key, color, label, linestyle="-", mark_best=False):
    points = [(e, v) for e, v in zip(history["epoch"], history[key]) if v is not None]
    if not points:
        return
    epochs, values = zip(*points)
    if mark_best:
        # el mejor checkpoint se elige por mAP@50:95 de validación; su valor va en la leyenda
        best = max(range(len(history["map"])), key=lambda i: (history["map"][i] or -1))
        value = history[key][best]
        if value is not None:
            label = f"{label} · mejor {value:.3f} (época {history['epoch'][best]})"
            ax.plot(history["epoch"][best], value, "o", markersize=8, color=color,
                    markeredgecolor="white", markeredgewidth=2, zorder=5)
    ax.plot(epochs, values, color=color, linewidth=2, linestyle=linestyle, label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="NOMBRE=RAÍZ")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/comparison"))
    args = parser.parse_args()

    runs = []
    for spec in args.run:
        name, _, root = spec.partition("=")
        if not root:
            parser.error(f"Usa NOMBRE=RAÍZ: {spec}")
        runs.append((name, Path(root)))
    if len(runs) > len(COLORS):
        parser.error(f"Como máximo {len(COLORS)} ejecuciones por figura.")

    data = {}
    for model in MODELS:
        for name, root in runs:
            path = root / model / name / "training.csv"
            if path.is_file():
                data[model, name] = read_history(path)
            else:
                print(f"Aviso: no existe {path}")
    models = [m for m in MODELS if any((m, n) in data for n, _ in runs)]
    if not models:
        raise SystemExit("No se encontró ningún training.csv.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_vs_".join(name for name, _ in runs)
    plt.rcParams["font.family"] = "DejaVu Sans"

    # figura 1: métricas de validación
    fig, axes = plt.subplots(len(models), 2, figsize=(11, 3.6 * len(models)), squeeze=False)
    for row, model in enumerate(models):
        for col, (key, label) in enumerate((("map", "mAP@50:95"), ("map50", "mAP@50"))):
            ax = axes[row][col]
            for (name, _), color in zip(runs, COLORS):
                if (model, name) in data:
                    plot_series(ax, data[model, name], key, color, name, mark_best=True)
            style(ax, f"{MODELS[model]} · {label} (val)", label)
            ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.text(0.01, 0.005, "Punto: mejor época por mAP@50:95 de validación (checkpoint guardado).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    metrics_path = args.output_dir / f"{suffix}_map.png"
    fig.savefig(metrics_path, dpi=200, facecolor="white")
    plt.close(fig)

    # figura 2: pérdidas, continua = train y discontinua = val
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 3.8), squeeze=False)
    for col, model in enumerate(models):
        ax = axes[0][col]
        for (name, _), color in zip(runs, COLORS):
            if (model, name) in data:
                plot_series(ax, data[model, name], "train_loss", color, f"{name} · train")
                plot_series(ax, data[model, name], "val_loss", color, f"{name} · val", linestyle="--")
        style(ax, f"{MODELS[model]} · pérdidas", "Pérdida")
        ax.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.005, "Las escalas de pérdida de cada modelo no son comparables entre sí.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    loss_path = args.output_dir / f"{suffix}_loss.png"
    fig.savefig(loss_path, dpi=200, facecolor="white")
    plt.close(fig)

    print(metrics_path)
    print(loss_path)


if __name__ == "__main__":
    main()
