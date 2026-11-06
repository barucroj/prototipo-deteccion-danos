"""End-to-end metrics of the A/B pipeline against a labeled batch.

For each image, the labeled damages are paired with the damages M3 detected by
box IoU (Hungarian, gated by ``min_iou``, the type does not have to match).
Each real damage then takes the state M4 gave to its detection:

- ``nuevo`` / ``preexistente`` / ``no_visible``: what M4 said.
- ``emparejado_mal``: a real preexisting damage that M4 called preexisting but
  paired with a detection that is not the same damage.
- ``sin_clasificar``: detected, but the pair failed the view check.
- ``no_detectado``: M3 found nothing over it.

A comparison entry none of whose detections matches a real damage is a
*false* detection with the state M4 gave it. A false ``nuevo`` is the costly
error: it would be charged to the customer.

Only new damages are measured, since they are what would be charged:

- Per damage: precision, recall and F1 of ``nuevo`` against everything else.
  A false positive is anything marked ``nuevo`` that is not a real new damage:
  a false detection or a real preexisting / not visible one. A false detection
  marked preexisting or not visible costs nothing and is not counted.
- Per pair: "does B have a new damage?". Accuracy needs negatives, so it is
  only meaningful with pairs without new damages. A pair is *exact* when all
  its new damages are found and nothing else is marked new.

The other states are kept as data (confusion matrix), without metrics.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.evaluation.ground_truth import PairGroundTruth
from src.m4_comparison.damages import NO_VISIBLE, NUEVO, PREEXISTENTE

DEFAULT_MIN_IOU = 0.3

EMPAREJADO_MAL = "emparejado_mal"
SIN_CLASIFICAR = "sin_clasificar"
NO_DETECTADO = "no_detectado"

REAL_STATES = (NUEVO, PREEXISTENTE, NO_VISIBLE)
PREDICTED_STATES = (NUEVO, PREEXISTENTE, NO_VISIBLE, EMPAREJADO_MAL, SIN_CLASIFICAR, NO_DETECTADO)
FALSE_STATES = (NUEVO, PREEXISTENTE, NO_VISIBLE, SIN_CLASIFICAR)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU of two ``[x1, y1, x2, y2]`` boxes."""
    width = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    height = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = ((box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
             + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - inter)
    return float(inter / union) if union > 0 else 0.0


def match_boxes(real: Sequence[Sequence[float]], detected: Sequence[Sequence[float]],
                min_iou: float) -> List[Tuple[int, int, float]]:
    """Pair real and detected boxes maximizing the number of pairs, then their IoU.

    Returns:
        ``(index_real, index_detected, iou)`` per pair with ``iou >= min_iou``.

    Same gating as :func:`src.m4_comparison.matching.match_points`: an invalid
    pair costs more than any sum of valid ones.
    """
    if len(real) == 0 or len(detected) == 0:
        return []
    ious = np.array([[box_iou(r, d) for d in detected] for r in real])
    invalid = ious < min_iou
    cost = 1.0 - ious
    forbidden = 1.0 + cost[~invalid].sum() if (~invalid).any() else 1.0
    rows, cols = linear_sum_assignment(np.where(invalid, forbidden, cost))
    return [(int(r), int(c), float(ious[r, c])) for r, c in zip(rows, cols) if not invalid[r, c]]


def evaluate_pair(truth: PairGroundTruth, report_a: dict, report_b: dict, comparison: dict,
                  min_iou: float = DEFAULT_MIN_IOU) -> dict:
    """Compare the M3 reports and the M4 report of one pair with its ground truth.

    Args:
        report_a / report_b: M3 reports (``danos`` with ``id``, ``tipo``, ``bbox``).
        comparison: M4 report; ``mapa_cambios`` is ``None`` for a blocked pair.
            The support detections it lists in ``histeresis`` are added to
            the detections of each image.

    The labeled boxes are brought to the pixels the detector saw (M1 may have
    reduced one photo to equalize the pair, see :func:`scale_boxes`), so
    ``bbox_real_a`` / ``bbox_real_b`` are in pixels of the analyzed images.

    Returns:
        ``{"par", "vista", "comparable", "danos": [...], "falsos": [...]}``,
        one entry per real damage and per false detection.
    """
    blocked = comparison["mapa_cambios"] is None
    # A support detection that M4 used as a partner (hysteresis) counts as a detection.
    support = comparison.get("histeresis") or {"a": [], "b": []}
    detected_a, detected_b = report_a["danos"] + support["a"], report_b["danos"] + support["b"]
    real_a = scale_boxes([x.bbox for x in truth.danos_a], truth.size_a, report_a)
    real_b = scale_boxes([x.bbox for x in truth.danos_b], truth.size_b, report_b)
    match_a = {r: (detected_a[d]["id"], iou) for r, d, iou in
               match_boxes(real_a, [x["bbox"] for x in detected_a], min_iou)}
    match_b = {r: (detected_b[d]["id"], iou) for r, d, iou in
               match_boxes(real_b, [x["bbox"] for x in detected_b], min_iou)}
    by_id_a = {e["id_a"]: e for e in comparison["danos"] if e["id_a"]}
    by_id_b = {e["id_b"]: e for e in comparison["danos"] if e["id_b"]}
    tipo_a = {d["id"]: d["tipo"] for d in detected_a}
    tipo_b = {d["id"]: d["tipo"] for d in detected_b}

    danos = []
    for real in truth.physical_damages():
        id_a, iou_a = match_a.get(real.index_a, (None, None)) if real.index_a is not None else (None, None)
        id_b, iou_b = match_b.get(real.index_b, (None, None)) if real.index_b is not None else (None, None)
        if id_b is not None:
            entry, tipo = by_id_b.get(id_b), tipo_b[id_b]
        elif id_a is not None:
            entry, tipo = by_id_a.get(id_a), tipo_a[id_a]
        else:
            entry, tipo = None, None

        if tipo is None:
            estado = NO_DETECTADO
        elif blocked:
            estado = SIN_CLASIFICAR
        else:
            estado = entry["estado"]
            wrong_partner = (entry["id_a"], entry["id_b"]) != (id_a, id_b)
            if real.estado == PREEXISTENTE and estado == PREEXISTENTE and wrong_partner:
                estado = EMPAREJADO_MAL
        danos.append({"id": real.id, "tipo": real.tipo, "estado_real": real.estado, "estado_predicho": estado,
                      "tipo_predicho": tipo, "id_a": id_a, "id_b": id_b,
                      "iou_a": _round(iou_a), "iou_b": _round(iou_b), **_real_boxes(real, real_a, real_b)})

    matched_a = {v[0] for v in match_a.values()}
    matched_b = {v[0] for v in match_b.values()}
    if blocked:
        candidates = ([{"id_a": d["id"], "id_b": None, "tipo": d["tipo"], "confianza": d["confianza"],
                        "estado": SIN_CLASIFICAR} for d in detected_a]
                      + [{"id_a": None, "id_b": d["id"], "tipo": d["tipo"], "confianza": d["confianza"],
                          "estado": SIN_CLASIFICAR} for d in detected_b])
    else:
        candidates = comparison["danos"]
    falsos = [{"id_a": e["id_a"], "id_b": e["id_b"], "tipo": e["tipo"], "confianza": e["confianza"],
               "estado_predicho": e["estado"]}
              for e in candidates if e["id_a"] not in matched_a and e["id_b"] not in matched_b]

    return {"par": truth.par, "vista": truth.vista, "comparable": not blocked, "danos": danos, "falsos": falsos}


def failed_pair(truth: PairGroundTruth, error: str) -> dict:
    """A pair the pipeline could not run (M1 rejected a photo): its damages stay unclassified.

    Nothing was analyzed, so the labeled boxes stay in pixels of the labeled images.
    """
    real_a = [list(map(float, x.bbox)) for x in truth.danos_a]
    real_b = [list(map(float, x.bbox)) for x in truth.danos_b]
    danos = [{"id": real.id, "tipo": real.tipo, "estado_real": real.estado, "estado_predicho": SIN_CLASIFICAR,
              "tipo_predicho": None, "id_a": None, "id_b": None, "iou_a": None, "iou_b": None,
              **_real_boxes(real, real_a, real_b)}
             for real in truth.physical_damages()]
    return {"par": truth.par, "vista": truth.vista, "comparable": False, "error": error,
            "danos": danos, "falsos": []}


def scale_boxes(boxes: Sequence[Sequence[float]], labeled_size: Optional[Tuple[int, int]],
                report: dict) -> List[List[float]]:
    """Labeled ``[x1, y1, x2, y2]`` boxes rescaled from the labeled image to the analyzed one.

    ``labeled_size`` is the (width, height) of the labeled image and the
    report's ``width`` / ``height`` the size M3 worked on. When either is
    missing, or both are equal, the boxes are returned unchanged.
    """
    boxes = [list(map(float, box)) for box in boxes]
    if labeled_size is None or "width" not in report or "height" not in report:
        return boxes
    sx, sy = report["width"] / labeled_size[0], report["height"] / labeled_size[1]
    if sx == 1 and sy == 1:
        return boxes
    return [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in boxes]


def _real_boxes(real, real_a: List[List[float]], real_b: List[List[float]]) -> dict:
    """Labeled boxes of a real damage in A and B, ``None`` where it is not visible."""
    return {"bbox_real_a": None if real.index_a is None else [round(v, 1) for v in real_a[real.index_a]],
            "bbox_real_b": None if real.index_b is None else [round(v, 1) for v in real_b[real.index_b]]}


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 3)


def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return round(2 * precision * recall / (precision + recall), 4)


def pair_outcome(pair: dict) -> dict:
    """New damages of one pair: is there one, was one flagged, and is the pair exact."""
    real_new = [d for d in pair["danos"] if d["estado_real"] == NUEVO]
    found = sum(d["estado_predicho"] == NUEVO for d in real_new)
    wrong = (sum(d["estado_real"] != NUEVO and d["estado_predicho"] == NUEVO for d in pair["danos"])
             + sum(f["estado_predicho"] == NUEVO for f in pair["falsos"]))
    return {"tiene_nuevo": bool(real_new), "marca_nuevo": found + wrong > 0,
            "exacto": found == len(real_new) and wrong == 0}


def summarize(pairs: List[dict]) -> dict:
    """Aggregate :func:`evaluate_pair` results into the metrics of the batch.

    Returns:
        ``nuevo`` (per damage: P, R, F1, TP, FP split into false detections and
        real non-new damages, FN), ``por_par`` (pair confusion, accuracy and
        exact pairs), and as data ``matriz`` (real state -> predicted state),
        ``falsos`` (false detections per predicted state), ``deteccion`` and
        ``tipos``. A metric with a zero denominator is ``None``.
    """
    matrix = {real: {pred: 0 for pred in PREDICTED_STATES} for real in REAL_STATES}
    falsos = Counter({state: 0 for state in FALSE_STATES})
    tipos = Counter()
    for pair in pairs:
        for dano in pair["danos"]:
            matrix[dano["estado_real"]][dano["estado_predicho"]] += 1
            if dano["tipo_predicho"] is not None:
                tipos[(dano["tipo"], dano["tipo_predicho"])] += 1
        for falso in pair["falsos"]:
            falsos[falso["estado_predicho"]] += 1

    tp = matrix[NUEVO][NUEVO]
    support = sum(matrix[NUEVO].values())
    fp_reales = sum(matrix[state][NUEVO] for state in REAL_STATES if state != NUEVO)
    fp = falsos[NUEVO] + fp_reales
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, support)
    nuevo = {"precision": precision, "recall": recall, "f1": _f1(precision, recall), "soporte": support,
             "tp": tp, "fp": fp, "fn": support - tp, "fp_falsos": falsos[NUEVO], "fp_reales": fp_reales,
             "falsos_por_par": _ratio(fp, len(pairs))}

    outcomes = [pair_outcome(pair) for pair in pairs]
    pair_tp = sum(o["tiene_nuevo"] and o["marca_nuevo"] for o in outcomes)
    pair_fp = sum(not o["tiene_nuevo"] and o["marca_nuevo"] for o in outcomes)
    pair_fn = sum(o["tiene_nuevo"] and not o["marca_nuevo"] for o in outcomes)
    pair_tn = len(outcomes) - pair_tp - pair_fp - pair_fn
    exactos = sum(o["exacto"] for o in outcomes)
    por_par = {"positivos": pair_tp + pair_fn, "negativos": pair_fp + pair_tn,
               "tp": pair_tp, "fp": pair_fp, "fn": pair_fn, "tn": pair_tn,
               "accuracy": _ratio(pair_tp + pair_tn, len(outcomes)),
               "exactos": exactos, "exactos_ratio": _ratio(exactos, len(outcomes))}

    reales = sum(sum(row.values()) for row in matrix.values())
    total_falsos = sum(falsos.values())
    detectados = reales - sum(matrix[state][NO_DETECTADO] for state in REAL_STATES)
    tipo_ok = sum(n for (real, pred), n in tipos.items() if real == pred)
    return {
        "pares": len(pairs),
        "pares_bloqueados": sum(not pair["comparable"] for pair in pairs),
        "nuevo": nuevo,
        "por_par": por_par,
        "matriz": matrix,
        "falsos": dict(falsos),
        "deteccion": {"reales": reales, "detectados": detectados, "recall": _ratio(detectados, reales),
                      "falsos": total_falsos, "falsos_por_par": _ratio(total_falsos, len(pairs))},
        "tipos": {"detectados": sum(tipos.values()), "coinciden": tipo_ok,
                  "accuracy": _ratio(tipo_ok, sum(tipos.values())),
                  "confusion": [{"real": real, "predicho": pred, "n": n}
                                for (real, pred), n in sorted(tipos.items())]},
    }
