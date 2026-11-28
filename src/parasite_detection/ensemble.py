"""fusión de detecciones de varios modelos con weighted boxes fusion (wbf)

implementación propia de solovyev et al. (2021), «weighted boxes fusion: ensembling boxes from different object
detection models». trabaja con predicciones en formato coco ([x, y, ancho, alto] en píxeles, ids de clase 1-8),
así que sirve igual para faster rcnn, yolo o cualquier modelo que guarde sus predicciones con ese formato

a diferencia del nms, que se queda con una caja y descarta las demás, wbf promedia las cajas que se solapan
ponderando por su confianza, y baja la confianza de las cajas que solo propone un modelo
"""

from collections import defaultdict

import numpy as np

MAX_DETECTIONS = 100


def _iou(box, boxes):
    # box: [x1, y1, x2, y2]; boxes: matriz n x 4 con el mismo formato
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def _fuse_one_class(boxes, scores, weights, iou_thr):
    """wbf de las cajas de una clase en una imagen

    boxes: n x 4 (x1, y1, x2, y2); scores: confianza ya multiplicada por el peso de su modelo
    """
    order = np.argsort(-scores)
    clusters = []      # índices de las cajas de cada grupo
    fused = []         # caja fusionada de cada grupo
    for index in order:
        if fused:
            ious = _iou(boxes[index], np.array(fused))
            best = int(np.argmax(ious))
            if ious[best] > iou_thr:
                clusters[best].append(index)
                members = clusters[best]
                # coordenadas: media ponderada por la confianza de las cajas del grupo
                fused[best] = (scores[members, None] * boxes[members]).sum(0) / scores[members].sum()
                continue
        clusters.append([index])
        fused.append(boxes[index].copy())
    results = []
    total_weight = sum(weights)
    for members, box in zip(clusters, fused):
        # confianza media del grupo, rebajada si lo proponen menos modelos de los que hay
        score = scores[members].mean() * min(len(weights), len(members)) / total_weight
        results.append((box, score))
    return results


def weighted_boxes_fusion(predictions_by_model, weights, iou_thr=0.55, skip_thr=0.0):
    """fusiona las predicciones coco de varios modelos

    predictions_by_model: lista de listas de predicciones coco, una por modelo
    weights: peso de cada modelo; puede ser una lista fija o una función image_id -> lista de pesos,
             que permite usar pesos distintos según el tipo de imagen
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for model_index, predictions in enumerate(predictions_by_model):
        for p in predictions:
            if p["score"] >= skip_thr:
                grouped[p["image_id"]][p["category_id"]].append((model_index, p["bbox"], p["score"]))

    fused_predictions = []
    for image_id, by_class in grouped.items():
        image_weights = weights(image_id) if callable(weights) else weights
        image_results = []
        for category_id, items in by_class.items():
            boxes = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for _, b, _ in items], dtype=float)
            scores = np.array([s * image_weights[m] for m, _, s in items], dtype=float)
            for box, score in _fuse_one_class(boxes, scores, image_weights, iou_thr):
                image_results.append({
                    "image_id": image_id, "category_id": category_id,
                    "bbox": [round(float(box[0]), 2), round(float(box[1]), 2),
                             round(float(box[2] - box[0]), 2), round(float(box[3] - box[1]), 2)],
                    "score": round(float(score), 5),
                })
        image_results.sort(key=lambda p: -p["score"])
        fused_predictions.extend(image_results[:MAX_DETECTIONS])
    return fused_predictions
