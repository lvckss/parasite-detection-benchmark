"""análisis de predicciones en formato coco: umbral, matriz de confusión, errores tipo tide y figuras

las funciones no dependen del modelo: sirven para faster rcnn y para yolo siempre que las predicciones
estén en formato coco ([x, y, ancho, alto] e ids de clase 1-8)
"""

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools.coco import COCO

from parasite_detection import mapviz as mv

# paleta de la guía de visualización: dos series categóricas, estados y tinta
BLUE, ORANGE = "#2a78d6", "#eb6834"
GOOD, CRITICAL, WARNING = "#0ca30c", "#d03b3b", "#fab219"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

ERROR_LABELS = {
    "cls": "Clasificación", "loc": "Localización", "dup": "Duplicado",
    "bg": "Fondo", "miss": "No detectado",
}


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 1.6, "figure.dpi": 90,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, path):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in (".jpg", ".jpeg"):
            # las rejillas de fotos pesan mucho menos en jpg y a menor resolución
            fig.savefig(path, dpi=130, pil_kwargs={"quality": 88})
        else:
            fig.savefig(path)
    return fig


def load_coco(data_dir, split):
    with redirect_stdout(StringIO()):
        return COCO(str(Path(data_dir) / "annotations" / f"instances_{split}.json"))


def load_predictions(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def class_names(coco):
    # nombre corto de cada especie para los ejes: el género basta para distinguirlas
    return {cat_id: cat["name"].split()[0] for cat_id, cat in coco.cats.items()}


def to_mapviz(coco, predictions):
    """convierte las anotaciones y predicciones coco al formato de listas de mapviz"""
    names = class_names(coco)
    xyxy = lambda b: (b[0], b[1], b[0] + b[2], b[1] + b[3])
    gts = [dict(img=a["image_id"], cls=names[a["category_id"]], box=xyxy(a["bbox"]), ann_id=a["id"])
           for a in coco.dataset["annotations"]]
    preds = [dict(img=p["image_id"], cls=names[p["category_id"]], box=xyxy(p["bbox"]), score=p["score"])
             for p in predictions]
    return gts, preds


# ----------------------------------------------------------------------------
# umbral de confianza
# ----------------------------------------------------------------------------
def threshold_sweep(coco, predictions, iou_thr=0.5, thresholds=np.arange(0.05, 0.96, 0.05)):
    """precisión, recall y f1 globales para cada umbral de confianza

    el emparejamiento voraz recorre las detecciones por confianza, así que basta con emparejar una vez:
    quedarse con las detecciones por encima de un umbral no cambia las parejas de las que sobreviven
    """
    gts, preds = to_mapviz(coco, predictions)
    status = mv.match(gts, preds, iou_thr)
    scores = np.array([p["score"] for p in preds])
    is_tp = np.array([s["status"] == "TP" for s in status])
    rows = []
    for thr in thresholds:
        keep = scores >= thr
        tp = int((is_tp & keep).sum())
        fp = int((~is_tp & keep).sum())
        fn = len(gts) - tp
        rows.append({
            "threshold": round(float(thr), 2), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(len(gts), 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# matriz de confusión
# ----------------------------------------------------------------------------
def confusion_matrix(coco, predictions, score_thr, iou_thr=0.5):
    """filas: clase real o fondo; columnas: clase predicha o no detectado

    cada detección por encima del umbral, de mayor a menor confianza, se queda con la caja real libre
    de la misma imagen con mayor iou (de cualquier clase) si supera iou_thr; si no, es fondo
    """
    cat_ids = sorted(coco.cats)
    index = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    n = len(cat_ids)
    matrix = np.zeros((n + 1, n + 1), dtype=int)
    by_image = {}
    for p in predictions:
        if p["score"] >= score_thr:
            by_image.setdefault(p["image_id"], []).append(p)
    for image_id in coco.getImgIds():
        anns = coco.imgToAnns[image_id]
        gt_boxes = [(a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3])
                    for a in anns]
        used = set()
        for p in sorted(by_image.get(image_id, []), key=lambda p: -p["score"]):
            b = p["bbox"]
            box = (b[0], b[1], b[0] + b[2], b[1] + b[3])
            best, best_iou = None, iou_thr
            for j, gt_box in enumerate(gt_boxes):
                value = mv.iou(box, gt_box)
                if j not in used and value >= best_iou:
                    best, best_iou = j, value
            if best is None:
                matrix[n, index[p["category_id"]]] += 1
            else:
                used.add(best)
                matrix[index[anns[best]["category_id"]], index[p["category_id"]]] += 1
        for j, a in enumerate(anns):
            if j not in used:
                matrix[index[a["category_id"]], n] += 1
    names = class_names(coco)
    labels = [names[c] for c in cat_ids]
    return pd.DataFrame(matrix, index=labels + ["Fondo"], columns=labels + ["No detectado"])


# ----------------------------------------------------------------------------
# errores tipo tide
# ----------------------------------------------------------------------------
def error_breakdown(coco, predictions, iou_thr=0.5):
    """desglose de mapviz: map@0.5 que se recuperaría si un oráculo corrigiera cada tipo de error"""
    gts, preds = to_mapviz(coco, predictions)
    result = mv.error_breakdown(gts, preds, iou_thr)
    table = pd.DataFrame({
        "error": [ERROR_LABELS[k] for k in ("cls", "loc", "dup", "bg", "miss")],
        "key": ["cls", "loc", "dup", "bg", "miss"],
        "count": [result["counts"][k] for k in ("cls", "loc", "dup", "bg", "miss")],
        "delta_map50": [result["delta"][k] for k in ("cls", "loc", "dup", "bg", "miss")],
    })
    return result, table


def image_errors(coco, predictions, score_thr, iou_thr=0.5):
    """estado de cada detección por encima del umbral y cajas reales sin detectar, por imagen"""
    kept = [p for p in predictions if p["score"] >= score_thr]
    gts, preds = to_mapviz(coco, kept)
    types, target, unexplained = mv.classify_errors(gts, preds, iou_thr)
    unexplained = set(unexplained)
    hit = {t for ty, t in zip(types, target) if ty == "TP"}
    per_image = {image_id: {"preds": [], "missed": []} for image_id in coco.getImgIds()}
    for p, ty, t in zip(preds, types, target):
        per_image[p["img"]]["preds"].append({**p, "type": ty, "target_cls": gts[t]["cls"] if t is not None else None})
    for j, g in enumerate(gts):
        if j not in hit:
            # explained=False: ninguna detección lo toca (no es un error de clase ni de localización)
            per_image[g["img"]]["missed"].append({**g, "explained": j not in unexplained})
    return per_image


# ----------------------------------------------------------------------------
# figuras
# ----------------------------------------------------------------------------
def plot_training(history, path=None):
    best = history.loc[history["map"].idxmax()]
    fig, (ax_loss, ax_map) = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax in (ax_loss, ax_map):
        ax.axvline(best["epoch"], color=AXIS, lw=1, ls="--", zorder=0)
        ax.set_xlabel("Época")
        # unas diez marcas: cada 2 épocas con 20 (faster rcnn), cada 6 con 60 (yolo)
        ax.set_xticks(history["epoch"][::max(1, len(history) // 10)])
    ax_loss.plot(history["epoch"], history["train_loss"], color=BLUE, marker="o", ms=4, label="Train")
    ax_loss.plot(history["epoch"], history["val_loss"], color=ORANGE, marker="o", ms=4, label="Validación")
    ax_loss.set_title("Pérdida por época")
    ax_loss.legend(loc="upper right")
    ax_map.plot(history["epoch"], history["map50"], color=BLUE, marker="o", ms=4, label="mAP@0.5")
    ax_map.plot(history["epoch"], history["map"], color=ORANGE, marker="o", ms=4, label="mAP@[.5:.95]")
    ax_map.set_title("mAP en validación")
    ax_map.set_ylim(0, 1)
    ax_map.legend(loc="lower right")
    ax_map.annotate(f"mejor época {int(best['epoch'])}\nmAP {best['map']:.3f} · mAP50 {best['map50']:.3f}",
                    (best["epoch"], best["map"]), xytext=(-12, -42), textcoords="offset points",
                    ha="right", fontsize=8.5, color=INK_2)
    fig.tight_layout()
    return save(fig, path)


def plot_pr_curves(pr_curves, per_class, title="", path=None):
    """una curva pr por especie (small multiples) a iou 0.5 y 0.75"""
    order = per_class.sort_values("ap50", ascending=False)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, sharey=True)
    for ax, (_, row) in zip(axes.flat, order.iterrows()):
        for iou, color in ((0.5, BLUE), (0.75, ORANGE)):
            curve = pr_curves[(pr_curves["category_id"] == row["category_id"]) & (pr_curves["iou"] == iou)]
            curve = curve[curve["precision"] >= 0]
            ax.plot(curve["recall"], curve["precision"], color=color, label=f"IoU {iou}")
        ax.set_title(f"{row['name'].split()[0]}  ·  AP50 {row['ap50']:.2f}", fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
    for ax in axes[1]:
        ax.set_xlabel("Recall")
    for ax in axes[:, 0]:
        ax.set_ylabel("Precisión")
    axes.flat[0].legend(loc="lower left")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save(fig, path)


def plot_ap_val_test(per_class_val, per_class_test, path=None):
    """ap por especie en validación y en test (dumbbell): muestra qué clases caen al pasar a test"""
    merged = per_class_test.merge(per_class_val, on=["category_id", "name"], suffixes=("_test", "_val"))
    merged = merged.sort_values("ap50_test")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = np.arange(len(merged))
    for ax, metric, title in ((axes[0], "ap50", "AP@0.5"), (axes[1], "ap", "AP@[.5:.95]")):
        ax.hlines(y, merged[f"{metric}_val"], merged[f"{metric}_test"], color=AXIS, lw=2, zorder=1)
        ax.scatter(merged[f"{metric}_val"], y, s=60, color=ORANGE, zorder=2, label="Validación",
                   edgecolor=SURFACE, linewidth=2)
        ax.scatter(merged[f"{metric}_test"], y, s=60, color=BLUE, zorder=3, label="Test",
                   edgecolor=SURFACE, linewidth=2)
        for yi, value in zip(y, merged[f"{metric}_test"]):
            ax.text(value, yi + 0.3, f"{value:.2f}", ha="center", fontsize=8, color=INK_2)
        ax.set_xlim(0.3, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"{n.split()[0]} ({o})" for n, o in zip(merged["name"], merged["objects_test"])])
    axes[0].tick_params(axis="y", colors=INK)
    axes[1].legend(loc="lower right")
    fig.text(0.01, -0.02, "Entre paréntesis, objetos de cada especie en test.", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return save(fig, path)


def plot_threshold_sweep(sweep, chosen, path=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(sweep["threshold"], sweep["precision"], color=BLUE, label="Precisión")
    ax.plot(sweep["threshold"], sweep["recall"], color=ORANGE, label="Recall")
    ax.plot(sweep["threshold"], sweep["f1"], color=INK, lw=2, label="F1")
    ax.axvline(chosen, color=AXIS, ls="--", lw=1, zorder=0)
    best = sweep.loc[sweep["threshold"] == chosen].iloc[0]
    ax.annotate(f"umbral {chosen:.2f} · F1 {best['f1']:.3f}", (chosen, best["f1"]),
                xytext=(-8, 16), textcoords="offset points", ha="right", fontsize=8.5, color=INK_2)
    ax.set_xlabel("Umbral de confianza")
    ax.set_ylim(0, 1.02)
    ax.set_title("Umbral de confianza en validación (IoU 0.5)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return save(fig, path)


def plot_confusion(matrix, score_thr, path=None):
    """colores normalizados por fila (qué fracción de cada clase real acaba en cada columna)"""
    values = matrix.to_numpy()
    fractions = values / np.maximum(values.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.2, 7))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    ax.imshow(fractions, cmap=cmap, vmin=0, vmax=1)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if values[i, j]:
                ax.text(j, i, values[i, j], ha="center", va="center", fontsize=9,
                        color="#ffffff" if fractions[i, j] > 0.55 else INK)
    ax.set_xticks(range(values.shape[1]), matrix.columns, rotation=40, ha="right")
    ax.set_yticks(range(values.shape[0]), matrix.index)
    ax.tick_params(colors=INK)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    n = len(matrix) - 1
    ax.axhline(n - 0.5, color=SURFACE, lw=3)
    ax.axvline(n - 0.5, color=SURFACE, lw=3)
    ax.set_title(f"Matriz de confusión en test (confianza ≥ {score_thr:.2f}, IoU ≥ 0.5)")
    fig.tight_layout()
    return save(fig, path)


def plot_error_breakdown(table, base_map50, title="", path=None):
    table = table.sort_values("delta_map50", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bars = ax.barh([f"{e} ({c})" for e, c in zip(table["error"], table["count"])], table["delta_map50"],
                   color=BLUE, height=0.6)
    for bar, value in zip(bars, table["delta_map50"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f"+{value:.3f}",
                va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, max(0.05, table["delta_map50"].max() * 1.25))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK)
    ax.set_xlabel("mAP@0.5 recuperado si un oráculo corrige ese tipo de error (entre paréntesis, nº de casos)")
    ax.set_title(title or f"¿Dónde se pierde el mAP? (mAP@0.5 base {base_map50:.3f})")
    fig.tight_layout()
    return save(fig, path)


def _label(ax, x, y, text, color, top):
    # la etiqueta va encima de la caja; si no cabe, se mete dentro para no tapar el título
    inside = y < top + 22
    ax.text(x, y + 2 if inside else y - 2, text, fontsize=7.5, color=INK if color == WARNING else "#ffffff",
            va="top" if inside else "bottom", clip_on=True,
            bbox=dict(boxstyle="square,pad=0.15", fc=color, ec="none"))


def draw_image(ax, image_dir, record, errors, title=None, focus=None, max_labels=8):
    """dibuja detecciones correctas (verde), erróneas (rojo) y cajas reales sin detectar (ámbar, discontinua)

    focus = (x1, y1, x2, y2) recorta la vista alrededor de esa zona; con más de max_labels cajas
    visibles se omiten los textos y el recuento pasa al título
    """
    with Image.open(Path(image_dir) / record["file_name"]) as file:
        ax.imshow(file.convert("RGB"))
    top = 0
    if focus is not None:
        margin = 0.6 * max(focus[2] - focus[0], focus[3] - focus[1]) + 30
        x1, y1 = max(0, focus[0] - margin), max(0, focus[1] - margin)
        x2, y2 = min(record["width"], focus[2] + margin), min(record["height"], focus[3] + margin)
        ax.set_xlim(x1, x2)
        ax.set_ylim(y2, y1)
        top = y1
        inside = lambda b: b[2] > x1 and b[0] < x2 and b[3] > y1 and b[1] < y2
    else:
        inside = lambda b: True
    visible = [g for g in errors["missed"] if inside(g["box"])] + [p for p in errors["preds"] if inside(p["box"])]
    labels = len(visible) <= max_labels
    for g in errors["missed"]:
        x1, y1, x2, y2 = g["box"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=WARNING, lw=2, ls="--"))
        if labels and inside(g["box"]):
            _label(ax, x1, y1, f"FN {g['cls']}", WARNING, top)
    for p in sorted(errors["preds"], key=lambda p: p["score"]):
        x1, y1, x2, y2 = p["box"]
        color = GOOD if p["type"] == "TP" else CRITICAL
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=2))
        if labels and inside(p["box"]):
            tag = "TP" if p["type"] == "TP" else ERROR_LABELS[p["type"]]
            extra = f" (real {p['target_cls']})" if p["type"] == "cls" else ""
            _label(ax, x1, y1, f"{tag}: {p['cls']} {p['score']:.2f}{extra}", color, top)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if not labels:
        tp = sum(p["type"] == "TP" for p in errors["preds"])
        fp = len(errors["preds"]) - tp
        counts = f"TP {tp} · FP {fp} · FN {len(errors['missed'])}"
        title = f"{title}\n{counts}" if title else counts
    if title:
        ax.set_title(title, fontsize=9.5)


def plot_examples(coco, image_dir, per_image, image_ids, titles, ncols=4, suptitle="", focus=None, path=None):
    nrows = int(np.ceil(len(image_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 4.3 * nrows), squeeze=False)
    focus = focus or [None] * len(image_ids)
    for ax, image_id, title, box in zip(axes.flat, image_ids, titles, focus):
        draw_image(ax, image_dir, coco.imgs[image_id], per_image[image_id], title, focus=box)
    for ax in list(axes.flat)[len(image_ids):]:
        ax.axis("off")
    handles = [
        Rectangle((0, 0), 1, 1, fill=False, ec=GOOD, lw=2, label="Detección correcta (TP)"),
        Rectangle((0, 0), 1, 1, fill=False, ec=CRITICAL, lw=2, label="Detección errónea (FP)"),
        Rectangle((0, 0), 1, 1, fill=False, ec=WARNING, lw=2, ls="--", label="Objeto no detectado (FN)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return save(fig, path)


def pick_correct_examples(coco, per_image):
    """una imagen por especie sin ningún error, con la mayor confianza media"""
    names = class_names(coco)
    chosen = {}
    for image_id, errors in per_image.items():
        anns = coco.imgToAnns[image_id]
        if not anns or errors["missed"] or any(p["type"] != "TP" for p in errors["preds"]):
            continue
        classes = {a["category_id"] for a in anns}
        if len(classes) != 1:
            continue
        name = names[classes.pop()]
        score = np.mean([p["score"] for p in errors["preds"]])
        if name not in chosen or score > chosen[name][1]:
            chosen[name] = (image_id, score)
    order = sorted(chosen)
    return [chosen[n][0] for n in order], order


def pick_error_examples(per_image):
    """para cada tipo de error, el caso de mayor confianza (para no detectado, el huevo más grande
    que ninguna detección toca)

    devuelve las imágenes, los tipos y la zona a ampliar en cada una
    """
    picks = {}
    for image_id, errors in per_image.items():
        for p in errors["preds"]:
            if p["type"] != "TP" and (p["type"] not in picks or p["score"] > picks[p["type"]][1]):
                picks[p["type"]] = (image_id, p["score"], p["box"])
        for g in [g for g in errors["missed"] if not g["explained"]]:
            area = (g["box"][2] - g["box"][0]) * (g["box"][3] - g["box"][1])
            if "miss" not in picks or area > picks["miss"][1]:
                picks["miss"] = (image_id, area, g["box"])
    keys = [k for k in ("cls", "loc", "bg", "dup", "miss") if k in picks]
    return [picks[k][0] for k in keys], keys, [picks[k][2] for k in keys]


def missed_by_class(coco, per_image, top=3):
    """cuántos huevos no detectados tiene cada especie y qué parte se concentra en sus peores imágenes"""
    rows = []
    for name in sorted(class_names(coco).values()):
        counts = pd.Series({i: sum(g["cls"] == name for g in e["missed"]) for i, e in per_image.items()})
        counts = counts[counts > 0].sort_values(ascending=False)
        rows.append({
            "especie": name, "no_detectados": int(counts.sum()), "imagenes": len(counts),
            f"en_top{top}": int(counts.head(top).sum()),
            f"top{top}_imagenes": ", ".join(f"{i} ({c})" for i, c in counts.head(top).items()),
        })
    return pd.DataFrame(rows).sort_values("no_detectados", ascending=False)


def worst_images(per_image, n=8):
    counts = {image_id: sum(p["type"] != "TP" for p in e["preds"]) + len(e["missed"])
              for image_id, e in per_image.items()}
    return sorted(counts, key=lambda k: (-counts[k], k))[:n], counts
