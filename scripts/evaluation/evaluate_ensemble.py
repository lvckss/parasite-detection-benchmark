"""ensemble de detectores con weighted boxes fusion, con pesos fijos o según el tipo de imagen

lee las predicciones coco que guardan evaluate_faster_rcnn.py y evaluate_yolo.py, así que no repite la
inferencia y sirve para cualquier combinación de ejecuciones (por ejemplo, los modelos mejorados del equipo).
todos los parámetros se eligen en validación; test solo se mide al final con los parámetros ya fijados

variantes que se comparan:
- wbf de cada modelo por separado: la fusión aplicada a un solo modelo (une sus cajas duplicadas)
- ensemble con pesos fijos: el mismo peso de cada modelo en todas las imágenes
- ensemble por tipo: el peso del segundo modelo depende de si el clasificador de tipo (entrenado con las
  etiquetas manuales del train) predice que la imagen es de microscopía o no
"""

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

from parasite_detection.analysis import load_coco
from parasite_detection.ensemble import weighted_boxes_fusion
from parasite_detection.evaluation import coco_evaluate, paired_bootstrap, save_results
from parasite_detection.image_types import group_of, leave_one_out_accuracy, predict_split_types
from parasite_detection.paths import DEDUP_DATA, RESULTS

# rejilla de búsqueda en validación
SECOND_WEIGHTS = [0.25, 0.5, 0.75, 1.0]
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.7]
SKIP_THRESHOLDS = [0.001, 0.01, 0.05]


def load_member(spec, split):
    model, run = spec.split(":", 1)
    directory = RESULTS / "metrics" / model / run
    return json.loads((directory / f"{split}_predictions.json").read_text(encoding="utf-8"))


def score(coco, predictions):
    evaluator, _ = coco_evaluate(coco, predictions, sorted(coco.imgs))
    return float(evaluator.stats[0]), float(evaluator.stats[1]), evaluator


def main():
    parser = argparse.ArgumentParser(description="Ensemble de detectores con WBF ajustado en validación.")
    parser.add_argument("--member", action="append", metavar="MODELO:RUN",
                        help="Primero el modelo de referencia. Por defecto fasterrcnn_v2:baseline y yolov10s:baseline.")
    parser.add_argument("--name", default="faster_yolo_baseline", help="Carpeta en results/metrics/ensemble/.")
    parser.add_argument("--data-dir", type=Path, default=DEDUP_DATA)
    parser.add_argument("--bootstrap", type=int, default=200, help="Remuestreos para el intervalo en test.")
    options = parser.parse_args()
    members = options.member or ["fasterrcnn_v2:baseline", "yolov10s:baseline"]
    if len(members) != 2:
        raise SystemExit("El ensemble por tipo admite exactamente dos modelos: referencia y segundo modelo.")
    out = RESULTS / "metrics/ensemble" / options.name
    out.mkdir(parents=True, exist_ok=True)

    coco = {s: load_coco(options.data_dir, s) for s in ("val", "test")}
    preds = {s: [load_member(m, s) for m in members] for s in ("val", "test")}

    # tipo de imagen predicho por el clasificador: es lo que el sistema tendría en un caso real
    accuracy, grouped_accuracy, _, _ = leave_one_out_accuracy(options.data_dir)
    types = {s: predict_split_types(s, options.data_dir) for s in ("val", "test")}
    groups = {s: {i: group_of(t) for i, (t, _) in types[s].items()} for s in ("val", "test")}
    print(f"Clasificador de tipo (train, dejando una fuera): {accuracy:.3f} en 4 tipos, "
          f"{grouped_accuracy:.3f} en microscopía frente al resto")

    rows = []

    def record(variant, params, val_map, val_map50):
        rows.append({"variante": variant, **params, "val_map": round(val_map, 4), "val_map50": round(val_map50, 4)})

    # 1. cada modelo solo, sin fusión
    for member, member_preds in zip(members, preds["val"]):
        record(f"solo {member}", {}, *score(coco["val"], member_preds)[:2])

    # 2. wbf de cada modelo por separado
    single_best = {}
    for index, member in enumerate(members):
        best = None
        for iou, skip in itertools.product(IOU_THRESHOLDS, SKIP_THRESHOLDS):
            fused = weighted_boxes_fusion([preds["val"][index]], [1.0], iou, skip)
            m, m50, _ = score(coco["val"], fused)
            record(f"wbf {member}", {"iou_thr": iou, "skip_thr": skip}, m, m50)
            if best is None or m > best[0]:
                best = (m, {"iou_thr": iou, "skip_thr": skip})
        single_best[member] = best[1]
        print(f"WBF {member}: val mAP {best[0]:.4f} con {best[1]}")

    # 3. ensemble con pesos fijos
    best_fixed = None
    for w, iou, skip in itertools.product(SECOND_WEIGHTS, IOU_THRESHOLDS, SKIP_THRESHOLDS):
        fused = weighted_boxes_fusion(preds["val"], [1.0, w], iou, skip)
        m, m50, _ = score(coco["val"], fused)
        record("ensemble fijo", {"peso_2": w, "iou_thr": iou, "skip_thr": skip}, m, m50)
        if best_fixed is None or m > best_fixed[0]:
            best_fixed = (m, {"peso_2": w, "iou_thr": iou, "skip_thr": skip})
    print(f"Ensemble fijo: val mAP {best_fixed[0]:.4f} con {best_fixed[1]}")

    # 4. ensemble por tipo: mismo iou y skip que el fijo, peso del segundo modelo según el grupo
    iou, skip = best_fixed[1]["iou_thr"], best_fixed[1]["skip_thr"]
    best_typed = None
    for w_micro, w_other in itertools.product(SECOND_WEIGHTS, repeat=2):
        weights = lambda image_id, g=groups["val"], a=w_micro, b=w_other: [1.0, a if g[image_id] == "microscopy" else b]
        fused = weighted_boxes_fusion(preds["val"], weights, iou, skip)
        m, m50, _ = score(coco["val"], fused)
        record("ensemble por tipo", {"peso_2_microscopia": w_micro, "peso_2_resto": w_other,
                                     "iou_thr": iou, "skip_thr": skip}, m, m50)
        if best_typed is None or m > best_typed[0]:
            best_typed = (m, {"peso_2_microscopia": w_micro, "peso_2_resto": w_other, "iou_thr": iou, "skip_thr": skip})
    print(f"Ensemble por tipo: val mAP {best_typed[0]:.4f} con {best_typed[1]}")

    with (out / "val_grid.csv").open("w", newline="", encoding="utf-8") as file:
        params = dict.fromkeys(k for r in rows for k in r if k not in ("variante", "val_map", "val_map50"))
        fields = ["variante", *params, "val_map", "val_map50"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # 5. test, una sola vez, con los parámetros elegidos en validación
    def build(split, variant):
        g = groups[split]
        if variant == "fijo":
            p = best_fixed[1]
            return weighted_boxes_fusion(preds[split], [1.0, p["peso_2"]], p["iou_thr"], p["skip_thr"])
        p = best_typed[1]
        weights = lambda i: [1.0, p["peso_2_microscopia"] if g[i] == "microscopy" else p["peso_2_resto"]]
        return weighted_boxes_fusion(preds[split], weights, p["iou_thr"], p["skip_thr"])

    summary = {"members": members, "type_classifier": {"loo_accuracy_4_types": accuracy,
                                                       "loo_accuracy_microscopy_vs_rest": grouped_accuracy},
               "selection": "map@[.5:.95] en validación", "chosen": {}, "val": {}, "test": {}}
    # cada variante se compara con el paso anterior: el ensemble fijo con el modelo de referencia,
    # y el ensemble por tipo con el fijo (lo único que cambia es usar el tipo de imagen)
    baselines = {"fijo": (preds["test"][0], members[0]), "tipo": (build("test", "fijo"), "ensemble_fijo")}
    for variant, chosen in (("fijo", best_fixed[1]), ("tipo", best_typed[1])):
        name = f"ensemble_{variant}"
        summary["chosen"][name] = chosen
        for split in ("val", "test"):
            start = time.perf_counter()
            fused = build(split, variant)
            # la fusión de todas las imágenes, para añadirla a la latencia de los modelos
            summary.setdefault("fusion_ms_por_imagen", {})[name] = (time.perf_counter() - start) * 1000 / len(coco[split].imgs)
            m, m50, evaluator = score(coco[split], fused)
            meta = {"model": name, "members": members, "split": split, "params": chosen,
                    "images": len(coco[split].imgs), "detections": len(fused),
                    "inference": {"note": "latencia: suma de los modelos, el clasificador de tipo y la fusión; "
                                          "ver results/metrics/latency/"}}
            save_results(out / name, split, meta, evaluator, coco[split], fused)
            summary[split][name] = {"map": m, "map50": m50}
        reference, reference_name = baselines[variant]
        ci = paired_bootstrap(coco["test"], build("test", variant), reference, sorted(coco["test"].imgs),
                              resamples=options.bootstrap)
        summary["test"][name][f"ic95_frente_a_{reference_name}"] = ci
        print(f"{name}: test mAP {summary['test'][name]['map']:.4f} | IC95 frente a {reference_name}: "
              f"[{ci['low']:+.4f}, {ci['high']:+.4f}]")
    for member, member_preds in zip(members, preds["test"]):
        summary["test"][member] = dict(zip(("map", "map50"), score(coco["test"], member_preds)[:2]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # tipo predicho de cada imagen, para el análisis por tipo
    with (out / "image_types.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "image_id", "tipo_predicho", "confianza", "grupo"])
        for split in ("val", "test"):
            for image_id, (t, c) in sorted(types[split].items()):
                writer.writerow([split, image_id, t, round(c, 3), group_of(t)])
    print(f"Resultados en {out}")


if __name__ == "__main__":
    main()
