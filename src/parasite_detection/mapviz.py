# copiado sin cambios del material de la práctica (P6/notebook_ap/ap_tide/mapviz.py)
# se usa para el desglose de errores tipo TIDE de la sección 9 del notebook p1_map_visual
"""mapviz: helpers for the P1 notebook "Reading mAP: the PR curve, its traps, and where the missing mAP goes".

Everything here is plain numpy + matplotlib (+ ipywidgets for the explorer), so it
runs on Colab, Kaggle, the cluster or a laptop with no GPU.

Data model (deliberately simple, no classes):
    GT   = dict(img=str, cls=str, box=(x1, y1, x2, y2))
    Pred = dict(img=str, cls=str, box=(x1, y1, x2, y2), score=float)
Boxes are in pixels, corner format, x2 > x1, y2 > y1 (same convention as torchvision).

For the toy images there is also a "world": the objects that are really in the image
(what gets painted). Normally world == GT; the data-fault experiments make them differ.
"""
from __future__ import annotations

import contextlib
import io
import math
import zlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

IMG_SIZE = (480, 320)  # (width, height) of the toy images

C_TP = "#2ca02c"
C_FP = "#d62728"
C_OFF = "#9e9e9e"
C_GT = "#ffffff"
C_MISS = "#ffbf00"
CLASS_COLORS = {"car": (0.30, 0.45, 0.70), "bike": (0.85, 0.55, 0.20), "person": (0.55, 0.35, 0.65)}


# ----------------------------------------------------------------------------
# Metric
# ----------------------------------------------------------------------------
def iou(a, b):
    """IoU of two boxes in (x1, y1, x2, y2) format."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def match(gts, preds, iou_thr=0.5):
    """Greedy matching, as done by VOC and COCO.

    Detections are visited from highest to lowest score. Each one takes the
    unmatched GT (same image, same class) with the highest IoU, provided that
    IoU >= iou_thr. A GT can be matched at most once; anything else is a FP.

    Returns a list aligned with `preds`: dict(status='TP'|'FP', gt=index|None,
    iou=best IoU with any same-class GT in the image, matched or not).
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
    used, out = set(), [None] * len(preds)
    for i in order:
        p = preds[i]
        best_j, best_v, best_any = None, -1.0, 0.0
        for j, g in enumerate(gts):
            if g["img"] != p["img"] or g["cls"] != p["cls"]:
                continue
            v = iou(p["box"], g["box"])
            best_any = max(best_any, v)
            if j not in used and v >= iou_thr and v > best_v:
                best_j, best_v = j, v
        if best_j is None:
            out[i] = dict(status="FP", gt=None, iou=best_any)
        else:
            used.add(best_j)
            out[i] = dict(status="TP", gt=best_j, iou=best_v)
    return out


def pr_curve(scores, is_tp, n_gt):
    """Precision/recall after each detection, detections sorted by score (desc)."""
    scores = np.asarray(scores, float)
    order = np.argsort(-scores, kind="mergesort")
    tp = np.asarray(is_tp, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / max(n_gt, 1)
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    return recall, precision


def _envelope(precision):
    precision = np.asarray(precision, float)
    if len(precision) == 0:
        return precision
    return np.maximum.accumulate(precision[::-1])[::-1]


def average_precision(recall, precision, method="coco101"):
    """Area under the PR curve, computed the standard way (101-point, as COCO and most libraries do).
    `method='all'` gives the VOC all-point variant; the difference is negligible here."""
    recall, precision = np.asarray(recall, float), np.asarray(precision, float)
    if len(recall) == 0:
        return 0.0
    if method == "all":
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = _envelope(np.concatenate([[0.0], precision, [0.0]]))
        i = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    env = _envelope(precision)
    vals = []
    for t in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, t, side="left")  # first index with recall >= t
        vals.append(env[idx] if idx < len(env) else 0.0)
    return float(np.mean(vals))


def _truncate_per_image(preds, max_dets):
    """Keep the top `max_dets` detections per (image, class), as COCOeval does."""
    keep, groups = [], {}
    for i, p in enumerate(preds):
        groups.setdefault((p["img"], p["cls"]), []).append(i)
    for idx in groups.values():
        keep.extend(sorted(idx, key=lambda i: -preds[i]["score"])[:max_dets])
    return [preds[i] for i in sorted(keep)]


def evaluate(gts, preds, iou_thr=0.5, max_dets=None):
    """Per-class AP and mAP (macro average over classes that have GT)."""
    if max_dets is not None:
        preds = _truncate_per_image(preds, max_dets)
    m = match(gts, preds, iou_thr)
    res = {}
    for c in sorted({g["cls"] for g in gts}):
        n_gt = sum(g["cls"] == c for g in gts)
        idx = [i for i, p in enumerate(preds) if p["cls"] == c]
        r, p = pr_curve([preds[i]["score"] for i in idx], [m[i]["status"] == "TP" for i in idx], n_gt)
        res[c] = average_precision(r, p)
    return dict(ap=res, mAP=float(np.mean(list(res.values()))) if res else 0.0)


def map50(gts, preds, max_dets=None):
    return evaluate(gts, preds, 0.5, max_dets)["mAP"]


def map_coco(gts, preds, max_dets=100):
    """mAP@[.5:.95]: AP averaged over the 10 IoU thresholds 0.50, 0.55, ..., 0.95."""
    return float(np.mean([evaluate(gts, preds, t, max_dets)["mAP"] for t in np.linspace(0.5, 0.95, 10)]))


def pooled_ap(gts, preds, iou_thr=0.5):
    """AP if you ignore classes when averaging: one PR curve over all boxes ("micro").
    Matching is still class-aware; only the averaging changes."""
    m = match(gts, preds, iou_thr)
    r, p = pr_curve([q["score"] for q in preds], [x["status"] == "TP" for x in m], len(gts))
    return average_precision(r, p)


def pycoco_eval(gts, preds, img_size=IMG_SIZE, max_dets=(1, 10, 100), verbose=False):
    """Run the official pycocotools COCOeval on our lists. Returns the main stats."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    imgs = sorted({g["img"] for g in gts} | {p["img"] for p in preds})
    img_id = {n: k + 1 for k, n in enumerate(imgs)}
    cats = sorted({g["cls"] for g in gts})
    cat_id = {c: k + 1 for k, c in enumerate(cats)}

    def xywh(b):
        return [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]

    ds = dict(
        images=[dict(id=img_id[n], width=img_size[0], height=img_size[1]) for n in imgs],
        categories=[dict(id=cat_id[c], name=c) for c in cats],
        annotations=[dict(id=k + 1, image_id=img_id[g["img"]], category_id=cat_id[g["cls"]],
                          bbox=xywh(g["box"]), area=xywh(g["box"])[2] * xywh(g["box"])[3], iscrowd=0)
                     for k, g in enumerate(gts)],
    )
    dets = [dict(image_id=img_id[p["img"]], category_id=cat_id[p["cls"]], bbox=xywh(p["box"]),
                 score=float(p["score"])) for p in preds if p["cls"] in cat_id]
    ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with ctx:
        coco_gt = COCO()
        coco_gt.dataset = ds
        coco_gt.createIndex()
        if not dets:
            return dict(mAP=0.0, mAP50=0.0, mAP75=0.0)
        E = COCOeval(coco_gt, coco_gt.loadRes(dets), "bbox")
        E.params.maxDets = list(max_dets)
        E.evaluate(); E.accumulate(); E.summarize()
    return dict(mAP=float(E.stats[0]), mAP50=float(E.stats[1]), mAP75=float(E.stats[2]))


# ----------------------------------------------------------------------------
# Toy data
# ----------------------------------------------------------------------------
def scene_a():
    """One image, one class ('car'), 6 GT boxes and 10 hand-placed detections."""
    G = [(30, 40, 130, 120), (170, 30, 250, 110), (300, 50, 420, 140),
         (40, 180, 150, 280), (200, 190, 280, 270), (340, 200, 380, 240)]
    P = [((32, 42, 128, 122), 0.95),   # near-perfect on car 0
         ((178, 38, 268, 122), 0.90),  # sloppy on car 1: TP@0.5, FP@0.75
         ((120, 240, 200, 310), 0.85), # confident background FP
         ((305, 55, 415, 135), 0.80),  # good on car 2
         ((36, 46, 124, 118), 0.70),   # duplicate on car 0
         ((45, 185, 140, 275), 0.62),  # good on car 3
         ((222, 206, 302, 290), 0.55), # borderline on car 4
         ((400, 15, 470, 75), 0.40),   # background FP
         ((350, 212, 395, 258), 0.30), # too sloppy on the small car 5
         ((205, 194, 283, 271), 0.10)] # good box on car 4, but very low score
    gts = [dict(img="scene_a", cls="car", box=b) for b in G]
    preds = [dict(img="scene_a", cls="car", box=b, score=s) for b, s in P]
    return gts, preds


def _place_boxes(rng, n, size, wh_range, existing):
    W, H = size
    out, tries = [], 0
    while len(out) < n and tries < 2000:
        tries += 1
        w = rng.uniform(*wh_range[0]); h = rng.uniform(*wh_range[1])
        x1 = rng.uniform(5, W - w - 5); y1 = rng.uniform(5, H - h - 5)
        b = (round(x1, 1), round(y1, 1), round(x1 + w, 1), round(y1 + h, 1))
        if all(iou(b, e) == 0 for e in existing + out):
            out.append(b)
    return out


def make_dataset(seed=0, n_imgs=4, cars=(3, 5), n_bikes=3, size=IMG_SIZE):
    """Several toy images with a frequent class ('car') and a rare one ('bike')."""
    rng = np.random.default_rng(seed)
    gts, bikes_left = [], n_bikes
    for k in range(n_imgs):
        img = f"img{k}"
        boxes = _place_boxes(rng, int(rng.integers(cars[0], cars[1] + 1)), size, ((50, 110), (35, 80)), [])
        gts += [dict(img=img, cls="car", box=b) for b in boxes]
        nb = bikes_left if k == n_imgs - 1 else int(rng.integers(0, min(2, bikes_left) + 1))
        bikes_left -= nb
        bb = _place_boxes(rng, nb, size, ((25, 45), (35, 60)), boxes)
        gts += [dict(img=img, cls="bike", box=b) for b in bb]
    return gts


def _jitter(rng, box, sigma, size):
    W, H = size
    w, h = box[2] - box[0], box[3] - box[1]
    b = np.array(box, float) + rng.normal(0, sigma, 4) * np.array([w, h, w, h])
    b[[0, 2]] = np.clip(b[[0, 2]], 0, W); b[[1, 3]] = np.clip(b[[1, 3]], 0, H)
    if b[2] - b[0] < 4: b[2] = b[0] + 4
    if b[3] - b[1] < 4: b[3] = b[1] + 4
    return tuple(np.round(b, 1))


def simulate_detector(world, seed=0, recall=None, jitter=0.06, score_tp=(0.75, 0.15),
                      score_fp=(0.35, 0.2), bg_per_img=1.5, dup_rate=0.1, dup_gap=(0.05, 0.3),
                      confuse_rate=0.0, size=IMG_SIZE):
    """A fake detector looking at the real objects (`world`): noisy boxes, background FPs, duplicates.

    recall       : dict cls -> probability that an object gets a detection at all
    jitter       : std of the box-corner noise, relative to the box size (localization quality)
    score_tp/fp  : (mean, std) of the confidence on objects / on background
    bg_per_img   : mean number of background detections per image
    dup_rate     : probability of a second box on the same object (what NMS should remove)
    confuse_rate : probability of predicting the wrong class
    """
    rng = np.random.default_rng(seed)
    recall = recall or {}
    classes = sorted({g["cls"] for g in world})
    preds = []
    for g in world:
        if rng.random() > recall.get(g["cls"], 0.9):
            continue
        cls = g["cls"]
        if confuse_rate and rng.random() < confuse_rate and len(classes) > 1:
            cls = str(rng.choice([c for c in classes if c != g["cls"]]))
        s = float(np.clip(rng.normal(*score_tp), 0.02, 0.99))
        preds.append(dict(img=g["img"], cls=cls, box=_jitter(rng, g["box"], jitter, size), score=s))
        if rng.random() < dup_rate:
            preds.append(dict(img=g["img"], cls=cls, box=_jitter(rng, g["box"], jitter * 1.5, size),
                              score=float(np.clip(s - rng.uniform(*dup_gap), 0.01, 0.99))))
    for img in sorted({g["img"] for g in world}):
        objs = [g["box"] for g in world if g["img"] == img]
        for _ in range(rng.poisson(bg_per_img)):
            b = _place_boxes(rng, 1, size, ((25, 100), (25, 80)), objs)
            if b:
                preds.append(dict(img=img, cls=str(rng.choice(classes)), box=b[0],
                                  score=float(np.clip(rng.normal(*score_fp), 0.01, 0.95))))
    for p in preds:  # break ties so every ranking is unambiguous
        p["score"] = round(min(0.999, max(0.001, p["score"] + rng.uniform(-1e-3, 1e-3))), 4)
    return preds


def annotate(world, seed=0, drop=0.0, jitter=0.0, swap=0.0, size=IMG_SIZE):
    """The labels a (possibly sloppy) annotator would produce for `world`.

    drop   : fraction of objects nobody labelled
    jitter : noise on the labelled boxes, relative to box size
    swap   : fraction of labels with the wrong class
    """
    rng = np.random.default_rng(seed + 1000)
    classes = sorted({g["cls"] for g in world})
    gts = []
    for g in world:
        if rng.random() < drop:
            continue
        q = dict(g)
        if jitter:
            q["box"] = _jitter(rng, g["box"], jitter, size)
        if swap and rng.random() < swap:
            q["cls"] = str(rng.choice([c for c in classes if c != g["cls"]]))
        gts.append(q)
    return gts


def scene_many_dets():
    """One image, 120 small objects, a detector that finds all of them but ranks 40 tiny
    background boxes above the last 40 real ones. Used for the pycocotools mismatch."""
    rng = np.random.default_rng(7)
    W, H = 960, 640
    gts, preds, k = [], [], 0
    for gy in range(10):
        for gx in range(12):
            x, y = 20 + gx * 78, 20 + gy * 62
            gts.append(dict(img="crowd", cls="bird", box=(x, y, x + 40, y + 34)))
            score = rng.uniform(0.5, 0.99) if k < 80 else rng.uniform(0.02, 0.2)
            preds.append(dict(img="crowd", cls="bird", box=(x + 2, y + 1, x + 41, y + 35), score=float(score)))
            k += 1
    for _ in range(40):
        x, y = rng.uniform(0, W - 30), rng.uniform(0, H - 30)
        preds.append(dict(img="crowd", cls="bird", box=(x, y, x + 3, y + 3), score=float(rng.uniform(0.25, 0.45))))
    return gts, preds, (W, H)


def box_with_iou(box, target_iou):
    """A box with the same centre whose IoU with `box` is exactly `target_iou` (shrunk)."""
    s = math.sqrt(target_iou)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = (box[2] - box[0]) * s, (box[3] - box[1]) * s
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def add_junk(preds, n=20, score=0.01, img=None, cls=None, seed=0, size=IMG_SIZE):
    """Return a copy of `preds` plus `n` random low-confidence boxes."""
    rng = np.random.default_rng(seed)
    img = img or preds[0]["img"]; cls = cls or preds[0]["cls"]
    junk = []
    for i in range(n):
        w, h = rng.uniform(20, 90), rng.uniform(20, 70)
        x, y = rng.uniform(0, size[0] - w), rng.uniform(0, size[1] - h)
        junk.append(dict(img=img, cls=cls, box=(x, y, x + w, y + h), score=score * (1 - i * 1e-4)))
    return [dict(p) for p in preds] + junk


# ----------------------------------------------------------------------------
# Fault injection for the error-analysis part
# ----------------------------------------------------------------------------
BASE_DETECTOR = dict(recall={"car": 0.95, "bike": 0.95}, jitter=0.03, score_tp=(0.8, 0.1),
                     score_fp=(0.3, 0.12), bg_per_img=0.5, dup_rate=0.03, confuse_rate=0.0)

# name -> (who is at fault, detector overrides, annotation settings)
FAULTS = {
    "clean":               ("-",     {}, {}),
    "sloppy box regression": ("model", dict(jitter=0.17), {}),
    "class confusion":     ("model", dict(confuse_rate=0.3), {}),
    "NMS too permissive":  ("model", dict(dup_rate=0.7, dup_gap=(0.0, 0.08)), {}),
    "confident background": ("model", dict(bg_per_img=2.0, score_fp=(0.85, 0.08)), {}),
    "low recall":          ("model", dict(recall={"car": 0.6, "bike": 0.6}), {}),
    "a bit of everything": ("model", dict(jitter=0.13, confuse_rate=0.12, dup_rate=0.25, dup_gap=(0.0, 0.15),
                                          bg_per_img=1.2, score_fp=(0.55, 0.2), recall={"car": 0.88, "bike": 0.8}), {}),
    "missing annotations": ("data",  {}, dict(drop=0.25)),
    "loose annotations":   ("data",  {}, dict(jitter=0.17)),
    "wrong class labels":  ("data",  {}, dict(swap=0.3)),
}


# Mystery detectors for the class: the fault is hidden on purpose (see the teacher guide).
_MYSTERY = {"A": ("missing annotations", 3), "B": ("loose annotations", 4), "C": ("sloppy box regression", 5)}


def mystery(name):
    """Return (world, gts, preds) for mystery detector 'A', 'B' or 'C'."""
    fault, seed = _MYSTERY[name]
    return make_case(fault, seed=seed)


def make_case(fault="clean", seed=0, n_imgs=8):
    """Return (world, gts, preds) for one fault. The detector sees `world`; it is scored against `gts`."""
    world = make_dataset(seed=seed, n_imgs=n_imgs, n_bikes=6)
    _, det_kw, ann_kw = FAULTS[fault]
    preds = simulate_detector(world, seed=seed + 1, **{**BASE_DETECTOR, **det_kw})
    gts = annotate(world, seed=seed, **ann_kw)
    return world, gts, preds


# ----------------------------------------------------------------------------
# Error breakdown (simplified TIDE)
# ----------------------------------------------------------------------------
ERROR_TYPES = ("loc", "cls", "dup", "bg")
ERROR_NAMES = dict(loc="localization", cls="classification", dup="duplicate", bg="background",
                   miss="missed GT")
ERROR_COLORS = dict(loc="#ff7f0e", cls="#9467bd", dup="#8c564b", bg="#d62728", miss=C_MISS)


def classify_errors(gts, preds, iou_thr=0.5, bg_thr=0.1):
    """Label every detection as 'TP' or one error type, plus the list of missed GT.

    loc : right class, overlaps an object but not enough (bg_thr <= IoU < iou_thr)
    cls : wrong class, but would be a TP for another class (IoU >= iou_thr)
    dup : right class, IoU >= iou_thr with an object that a higher-scored box already took
    bg  : overlaps no object (IoU < bg_thr with everything)
    missed : GT with no TP and not explained by a loc/cls error
    """
    m = match(gts, preds, iou_thr)
    types, target = [], []
    for p, mm in zip(preds, m):
        if mm["status"] == "TP":
            types.append("TP"); target.append(mm["gt"]); continue
        same = [(iou(p["box"], g["box"]), j) for j, g in enumerate(gts)
                if g["img"] == p["img"] and g["cls"] == p["cls"]]
        other = [(iou(p["box"], g["box"]), j) for j, g in enumerate(gts)
                 if g["img"] == p["img"] and g["cls"] != p["cls"]]
        bs = max(same, default=(0.0, None)); bo = max(other, default=(0.0, None))
        if bs[0] >= iou_thr:
            types.append("dup"); target.append(bs[1])
        elif bo[0] >= iou_thr:
            types.append("cls"); target.append(bo[1])
        elif max(bs[0], bo[0]) >= bg_thr:
            types.append("loc"); target.append(bs[1] if bs[0] >= bo[0] else bo[1])
        else:
            types.append("bg"); target.append(None)
    hit = {mm["gt"] for mm in m if mm["status"] == "TP"}
    explained = {t for ty, t in zip(types, target) if ty in ("loc", "cls")}
    missed = [j for j in range(len(gts)) if j not in hit and j not in explained]
    return types, target, missed


def _fix(gts, preds, types, target, kind):
    out = []
    taken = {t for ty, t in zip(types, target) if ty == "TP"}
    for p, ty, t in zip(preds, types, target):
        q = dict(p)
        if ty == kind:
            if kind in ("dup", "bg"):
                continue
            if t in taken:
                continue  # fixing it would only create a duplicate: suppress it
            if kind == "cls":
                q["cls"] = gts[t]["cls"]
            if kind == "loc":
                q["box"] = gts[t]["box"]; q["cls"] = gts[t]["cls"]
            taken.add(t)
        out.append(q)
    return out


def error_breakdown(gts, preds, iou_thr=0.5):
    """mAP gained if each error type were fixed by an oracle, one type at a time."""
    types, target, missed = classify_errors(gts, preds, iou_thr)
    base = evaluate(gts, preds, iou_thr)["mAP"]
    delta = {k: evaluate(gts, _fix(gts, preds, types, target, k), iou_thr)["mAP"] - base for k in ERROR_TYPES}
    counts = {k: types.count(k) for k in ERROR_TYPES}
    counts["miss"] = len(missed)
    lo = min([p["score"] for p in preds], default=0.5)
    fixed = [dict(p) for p in preds] + [dict(img=gts[j]["img"], cls=gts[j]["cls"], box=gts[j]["box"],
                                             score=lo * 0.5) for j in missed]
    delta["miss"] = evaluate(gts, fixed, iou_thr)["mAP"] - base
    return dict(base=base, delta=delta, counts=counts, types=types, target=target, missed=missed)


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def make_canvas(objects, img, size=IMG_SIZE):
    """A synthetic 'photo': noisy background with the objects painted as blobs."""
    W, H = size
    rng = np.random.default_rng(zlib.crc32(img.encode()))
    yy, xx = np.mgrid[0:H, 0:W]
    base = 0.18 + 0.10 * (yy / H) + 0.04 * np.sin(xx / 37.0)
    canvas = np.stack([base, base * 1.05, base * 1.15], -1) + rng.normal(0, 0.015, (H, W, 3))
    for g in objects:
        if g["img"] != img:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in g["box"]]
        col = np.array(CLASS_COLORS.get(g["cls"], (0.5, 0.6, 0.5)))
        canvas[y1:y2, x1:x2] = col * rng.uniform(0.8, 1.1)
        dx, dy = max(1, (x2 - x1) // 5), max(1, (y2 - y1) // 4)
        canvas[y1 + dy:y2 - dy, x1 + dx:x2 - dx] = col * 0.55
    return np.clip(canvas, 0, 1)


def _rank_labels(preds):
    """Rank 1 = highest score, per class."""
    labels = {}
    for c in {p["cls"] for p in preds}:
        idx = sorted([i for i, p in enumerate(preds) if p["cls"] == c], key=lambda i: -preds[i]["score"])
        for r, i in enumerate(idx, 1):
            labels[i] = r
    return labels


def _label(ax, placed, x1, y1, y2, text, col, alpha=1.0, fs=7.5):
    """Label above the box, or below it if another label is already there."""
    above = all(abs(x1 - px) > 50 or abs(y1 - py) > 14 for px, py in placed)
    ly, va = (y1 - 2, "bottom") if above else (y2 + 2, "top")
    placed.append((x1, y1 if above else y2 + 16))
    ax.text(x1 + 1, ly, text, color="white", fontsize=fs, va=va, clip_on=True,
            bbox=dict(fc=col, ec="none", alpha=0.85 * alpha, pad=1.2))


def draw_scene(ax, gts, preds, img=None, matches=None, kept=None, image=None, world=None,
               size=IMG_SIZE, show_scores=True, cls=None):
    """Draw one image: GT in white (amber dashed if missed), detections coloured by status.

    matches : output of match(gts, preds, thr), or None to draw everything neutral
    kept    : boolean list aligned with preds (False = below the confidence threshold)
    world   : objects to paint (defaults to the GT)
    """
    img = img or (gts[0]["img"] if gts else preds[0]["img"])
    ax.imshow(make_canvas(world if world is not None else gts, img, size) if image is None else image)
    kept = kept if kept is not None else [True] * len(preds)
    ranks = _rank_labels(preds)
    multi = len({q["cls"] for q in preds}) > 1 and not cls
    hit = set()
    if matches is not None:
        hit = {m["gt"] for m, k in zip(matches, kept) if k and m["status"] == "TP"}
    for j, g in enumerate(gts):
        if g["img"] != img or (cls and g["cls"] != cls):
            continue
        missed = matches is not None and j not in hit
        x1, y1, x2, y2 = g["box"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2.2,
                               ec=C_MISS if missed else C_GT, ls="--" if missed else "-"))
    placed = []
    for i in sorted(range(len(preds)), key=lambda i: -preds[i]["score"]):
        p = preds[i]
        if p["img"] != img or (cls and p["cls"] != cls):
            continue
        x1, y1, x2, y2 = p["box"]
        if not kept[i]:
            col, ls, lw, alpha = C_OFF, ":", 1.2, 0.8
        elif matches is None:
            col, ls, lw, alpha = "#4fc3f7", "-", 1.8, 1.0
        else:
            col, ls, lw, alpha = (C_TP if matches[i]["status"] == "TP" else C_FP), "-", 2.0, 1.0
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=lw, ec=col, ls=ls, alpha=alpha))
        if show_scores:
            _label(ax, placed, x1, y1, y2, f"#{ranks[i]} {p['score']:.2f}" + (f" {p['cls']}" if multi else ""),
                   col, alpha)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def show_scene(gts, preds, img=None, iou_thr=None, size=IMG_SIZE, figsize=(7, 4.8), cls=None):
    """Static view. With iou_thr=None, detections are not coloured."""
    fig, ax = plt.subplots(figsize=figsize)
    m = match(gts, preds, iou_thr) if iou_thr is not None else None
    draw_scene(ax, gts, preds, img=img, matches=m, size=size, cls=cls)
    if iou_thr is None:
        ax.set_title("white = ground truth, blue = detections (#rank score)")
    else:
        ax.set_title(f"IoU threshold {iou_thr}: green = TP, red = FP, amber dashed = missed GT")
    plt.tight_layout(); plt.show()


def _pr_table_html(rows, k):
    head = ("<tr><th>rank</th><th>score</th><th>best IoU</th><th>status</th>"
            "<th>cum TP</th><th>cum FP</th><th>precision</th><th>recall</th></tr>")
    body = []
    for n, r in enumerate(rows, 1):
        style = "opacity:0.35;" if n > k else ""
        if n == k:
            style += "outline:2px solid #1f77b4;"
        col = C_TP if r["status"] == "TP" else C_FP
        body.append(
            f"<tr style='{style}'><td>#{n}</td><td>{r['score']:.2f}</td><td>{r['iou']:.2f}</td>"
            f"<td style='color:{col};font-weight:bold'>{r['status']}</td><td>{r['ctp']}</td><td>{r['cfp']}</td>"
            f"<td>{r['p']:.3f}</td><td>{r['r']:.3f}</td></tr>")
    return ("<table style='font-family:monospace;font-size:12px;border-collapse:collapse' "
            "cellpadding='3'>" + head + "".join(body) + "</table>")


def _draw_pr(ax, r, p, k, statuses, title_extra=""):
    env = _envelope(p)
    rr = np.concatenate([[0], r]); ee = np.concatenate([[env[0] if len(env) else 0], env])
    ax.fill_between(rr, 0, ee, step="pre", color="#1f77b4", alpha=0.12, label="area = AP")
    ax.plot(r, p, "-", color="#bbbbbb", lw=1, zorder=1)
    ax.plot(r[:k], p[:k], "-", color="#333333", lw=1.6, zorder=2)
    for n in range(len(r)):
        col = C_TP if statuses[n] == "TP" else C_FP
        ax.plot(r[n], p[n], "o", ms=6, color=col, alpha=1.0 if n < k else 0.25, zorder=3)
    if k > 0:
        ax.plot(r[k - 1], p[k - 1], "D", ms=13, mfc="none", mec="#1f77b4", mew=2.2, zorder=4,
                label=f"operating point: P={p[k-1]:.2f}, R={r[k-1]:.2f}")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.grid(alpha=0.3)
    ax.set_title(f"PR curve: AP = {average_precision(r, p):.3f} {title_extra}")
    ax.legend(loc="lower left", fontsize=8)


def pr_explorer(gts, preds, iou_thr=0.5, size=IMG_SIZE):
    """Interactive linked view: image, PR curve and ranked table.

    "top-k kept" is the confidence threshold seen from the ranking side: keeping the
    top-k detections == threshold at the k-th score. Press play to watch the PR curve
    being drawn one detection at a time.
    """
    import ipywidgets as W
    from IPython.display import display, HTML

    classes = sorted({g["cls"] for g in gts})
    imgs = sorted({g["img"] for g in gts} | {p["img"] for p in preds})
    cls_dd = W.Dropdown(options=classes, value=classes[0], description="class")
    img_dd = W.Dropdown(options=imgs, value=imgs[0], description="image")
    iou_sl = W.FloatSlider(value=iou_thr, min=0.05, max=0.95, step=0.05, description="IoU thr",
                           continuous_update=False, readout_format=".2f")
    n0 = sum(p["cls"] == classes[0] for p in preds)
    k_sl = W.IntSlider(value=n0, min=0, max=n0, description="top-k kept", continuous_update=False)
    play = W.Play(value=n0, min=0, max=n0, interval=900)
    W.link((play, "value"), (k_sl, "value"))  # kernel-side link: works on Colab too

    def on_cls(change):
        n = sum(p["cls"] == change["new"] for p in preds)
        k_sl.max = play.max = n
        k_sl.value = n
    cls_dd.observe(on_cls, names="value")

    def render(c, im, t, k):
        P = [p for p in preds if p["cls"] == c]
        G = [g for g in gts if g["cls"] == c]
        m = match(G, P, t)
        order = sorted(range(len(P)), key=lambda i: -P[i]["score"])
        r, p = pr_curve([P[i]["score"] for i in order], [m[i]["status"] == "TP" for i in order], len(G))
        kept = [False] * len(P)
        for i in order[:k]:
            kept[i] = True
        thr = P[order[k - 1]]["score"] if k > 0 else 1.0
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
        draw_scene(a1, G, P, img=im, matches=m, kept=kept, size=size)
        a1.set_title(f"{im}: keeping score >= {thr:.2f} | green TP, red FP, grey = dropped, amber = missed")
        _draw_pr(a2, r, p, k, [m[i]["status"] for i in order], f"(IoU >= {t:.2f}, {len(G)} GT)")
        plt.tight_layout(); plt.show()
        rows, ctp, cfp = [], 0, 0
        for n, i in enumerate(order):
            ctp += m[i]["status"] == "TP"; cfp += m[i]["status"] == "FP"
            rows.append(dict(score=P[i]["score"], iou=m[i]["iou"], status=m[i]["status"],
                             ctp=ctp, cfp=cfp, p=p[n], r=r[n]))
        display(HTML(_pr_table_html(rows, k)))

    out = W.interactive_output(render, dict(c=cls_dd, im=img_dd, t=iou_sl, k=k_sl))
    top = [iou_sl, W.HBox([play, k_sl])]
    if len(classes) > 1:
        top.insert(0, cls_dd)
    if len(imgs) > 1:
        top.insert(0, img_dd)
    display(W.VBox([W.HBox(top[:2]), W.HBox(top[2:])]), out)


def plot_pr_compare(gts, models, iou_thr=0.5):
    """Overlay the PR curves of several prediction lists (dict name -> preds), single class."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name, P in models.items():
        m = match(gts, P, iou_thr)
        r, p = pr_curve([q["score"] for q in P], [x["status"] == "TP" for x in m], len(gts))
        ax.plot(r, p, "o-", ms=4, lw=1.6, label=f"{name}: AP = {average_precision(r, p):.3f}")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3)
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.legend(loc="lower left")
    plt.tight_layout(); plt.show()


def plot_per_class(gts, preds, iou_thr=0.5):
    """One PR curve per class + the mAP vs. pooled-AP comparison."""
    classes = sorted({g["cls"] for g in gts})
    m = match(gts, preds, iou_thr)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    aps = {}
    for c in classes:
        idx = [i for i, p in enumerate(preds) if p["cls"] == c]
        n_gt = sum(g["cls"] == c for g in gts)
        r, p = pr_curve([preds[i]["score"] for i in idx], [m[i]["status"] == "TP" for i in idx], n_gt)
        aps[c] = average_precision(r, p)
        ax.plot(r, p, "o-", ms=4, lw=1.8, color=CLASS_COLORS.get(c), label=f"{c}: {n_gt} GT, AP = {aps[c]:.3f}")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(f"mAP@{iou_thr} = {np.mean(list(aps.values())):.3f}   |   "
                 f"pooled (class-blind average) AP = {pooled_ap(gts, preds, iou_thr):.3f}")
    ax.legend(loc="lower left"); plt.tight_layout(); plt.show()


def plot_ap_vs_iou(gts, models, thresholds=None):
    """AP as a function of the IoU threshold, one line per model (dict name -> preds)."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, P in models.items():
        aps = [evaluate(gts, P, t)["mAP"] for t in thresholds]
        axes[0].plot(thresholds, aps, "o-", label=f"{name}: mAP@.5 = {map50(gts, P):.3f}, "
                                                    f"mAP@[.5:.95] = {map_coco(gts, P):.3f}")
        ious = [x["iou"] for x in match(gts, P, 0.1) if x["status"] == "TP"]
        axes[1].hist(ious, bins=np.linspace(0, 1, 21), alpha=0.55, label=name)
    axes[0].axvspan(0.5, 0.95, color="#ffbf00", alpha=0.12, label="COCO range [.5:.95]")
    axes[0].set_xlabel("IoU threshold"); axes[0].set_ylabel("mAP"); axes[0].set_ylim(-0.02, 1.05)
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_title("mAP as a function of the IoU threshold")
    axes[1].set_xlabel("IoU of each detection with its object"); axes[1].set_ylabel("# detections")
    axes[1].set_title("Localization quality"); axes[1].legend()
    plt.tight_layout(); plt.show()


def _bars(ax, res, xmax=None, title=""):
    keys = list(ERROR_TYPES) + ["miss"]
    vals = [res["delta"][k] for k in keys]
    bars = ax.barh([f"{ERROR_NAMES[k]} ({res['counts'][k]})" for k in keys], vals,
                   color=[ERROR_COLORS[k] for k in keys])
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.004, b.get_y() + b.get_height() / 2, f"+{v:.2f}", va="center", fontsize=8)
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, xmax or max(0.05, max(vals) * 1.25))
    ax.set_title(title or f"base mAP@.5 = {res['base']:.2f}", fontsize=10)


def plot_error_breakdown(res, title=""):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _bars(ax, res, title=title or f"Where is the missing mAP? (base mAP@.5 = {res['base']:.3f})")
    ax.set_xlabel("mAP gained if an oracle fixed this error type (count in brackets)")
    plt.tight_layout(); plt.show()


def plot_fingerprints(cases, ncols=3):
    """Small multiples: one error breakdown per case (dict name -> breakdown result)."""
    n = len(cases); nrows = math.ceil(n / ncols)
    xmax = max(max(r["delta"].values()) for r in cases.values()) * 1.3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.9 * nrows), squeeze=False)
    for ax, (name, res) in zip(axes.flat, cases.items()):
        _bars(ax, res, xmax, f"{name}\nmAP@.5 = {res['base']:.2f}")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    plt.tight_layout(); plt.show()


def plot_error_scores(preds, res):
    """Score of every detection, grouped by what it turned out to be."""
    keys = ["TP"] + list(ERROR_TYPES)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    rng = np.random.default_rng(0)
    for y, k in enumerate(keys):
        s = [p["score"] for p, t in zip(preds, res["types"]) if t == k]
        ax.scatter(s, y + rng.uniform(-0.18, 0.18, len(s)), s=22, alpha=0.8,
                   color=C_TP if k == "TP" else ERROR_COLORS[k])
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(["TP"] + [ERROR_NAMES[k] for k in ERROR_TYPES])
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("confidence score"); ax.grid(axis="x", alpha=0.3)
    ax.set_title("Where each error sits in the ranking")
    plt.tight_layout(); plt.show()


def draw_errors(ax, gts, preds, res, img, image=None, world=None, size=IMG_SIZE, min_score=0.0):
    """Image with every detection coloured by its error type and missed GT in amber."""
    ax.imshow(make_canvas(world if world is not None else gts, img, size) if image is None else image)
    for j, g in enumerate(gts):
        if g["img"] != img:
            continue
        x1, y1, x2, y2 = g["box"]
        miss = j in res["missed"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2.2,
                               ec=C_MISS if miss else C_GT, ls="--" if miss else "-"))
    placed = []
    for i in sorted(range(len(preds)), key=lambda i: -preds[i]["score"]):
        p, ty = preds[i], res["types"][i]
        if p["img"] != img or p["score"] < min_score:
            continue
        x1, y1, x2, y2 = p["box"]
        col = C_TP if ty == "TP" else ERROR_COLORS[ty]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.8, ec=col))
        _label(ax, placed, x1, y1, y2, f"{ty} {p['cls']} {p['score']:.2f}", col, fs=7)
    ax.set_xticks([]); ax.set_yticks([])


def worst_images(gts, preds, res, n=2):
    """Images with the most errors (FPs + missed GT)."""
    bad = {}
    for p, t in zip(preds, res["types"]):
        bad[p["img"]] = bad.get(p["img"], 0) + (t != "TP")
    for j in res["missed"]:
        bad[gts[j]["img"]] = bad.get(gts[j]["img"], 0) + 1
    return sorted(bad, key=lambda k: (-bad[k], k))[:n]


def show_errors(gts, preds, res, imgs, world=None, image=None, size=IMG_SIZE, ncols=2, min_score=0.0):
    """draw_errors for several images. Legend: white GT, amber dashed missed, colours = error types."""
    nrows = math.ceil(len(imgs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.7 * nrows), squeeze=False)
    for ax, img in zip(axes.flat, imgs):
        draw_errors(ax, gts, preds, res, img, image=None if image is None else image[img], world=world,
                    size=size, min_score=min_score)
        ax.set_title(img)
    for ax in list(axes.flat)[len(imgs):]:
        ax.axis("off")
    plt.tight_layout(); plt.show()


def show_error_gallery(gts, preds, res, world=None, size=IMG_SIZE):
    """One zoomed example of each error type (the highest-scored one)."""
    keys = list(ERROR_TYPES) + ["miss"]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.3 * len(keys), 3.3))
    for ax, k in zip(axes, keys):
        ax.set_xticks([]); ax.set_yticks([])
        if k == "miss":
            cand = [(0, j) for j in res["missed"]]
        else:
            cand = [(-p["score"], i) for i, (p, t) in enumerate(zip(preds, res["types"])) if t == k]
        if not cand:
            ax.set_title(f"{ERROR_NAMES[k]}\n(none)"); ax.axis("off"); continue
        _, i = min(cand)
        if k == "miss":
            g = gts[i]; img = g["img"]; boxes = [g["box"]]
        else:
            p = preds[i]; img = p["img"]; boxes = [p["box"]]
            t = res["target"][i]
            if t is not None:
                boxes.append(gts[t]["box"])
        x1 = max(0, min(b[0] for b in boxes) - 40); y1 = max(0, min(b[1] for b in boxes) - 40)
        x2 = min(size[0], max(b[2] for b in boxes) + 40); y2 = min(size[1], max(b[3] for b in boxes) + 40)
        draw_errors(ax, gts, preds, res, img, world=world, size=size)
        ax.set_xlim(x1, x2); ax.set_ylim(y2, y1)
        ax.set_title(ERROR_NAMES[k], color=ERROR_COLORS[k], fontweight="bold")
    plt.tight_layout(); plt.show()
