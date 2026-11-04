"""Turn a batch evaluation (``src.evaluation``) into what the web interface shows. No Streamlit, no torch."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.app.view import STATE_LABELS, damage_label
from src.evaluation.metrics import FALSE_STATES, NO_DETECTADO, PREDICTED_STATES, REAL_STATES, pair_outcome
from src.m4_comparison.damages import NUEVO

#: Evaluation state -> label shown to the user.
EVALUATION_LABELS = {**STATE_LABELS, "emparejado_mal": "Mal emparejado", "sin_clasificar": "Sin clasificar",
                     "no_detectado": "No detectado"}
#: Color of the labeled (ground truth) boxes drawn over the change map, BGR.
GROUND_TRUTH_COLOR = (0, 200, 0)


def state_label(state: str) -> str:
    return EVALUATION_LABELS.get(state, state)


def matrix_rows(summary: dict) -> List[dict]:
    """Confusion matrix: one row per real state with damages, plus the false detections."""
    rows = []
    for state in REAL_STATES:
        row = summary["matriz"][state]
        if sum(row.values()):
            rows.append({"Real": state_label(state), **{state_label(p): row[p] for p in PREDICTED_STATES}})
    rows.append({"Real": "(falso)", **{state_label(p): summary["falsos"].get(p, 0) if p in FALSE_STATES else None
                                       for p in PREDICTED_STATES}})
    return rows


def pair_rows(pairs: List[dict]) -> List[dict]:
    """One row per pair of the batch."""
    rows = []
    for pair in pairs:
        danos = pair["danos"]
        outcome = pair_outcome(pair)
        rows.append({
            "Par": pair["par"],
            "Vista": pair["vista"],
            "Comparado": "error" if "error" in pair else ("Sí" if pair["comparable"] else "No"),
            "Nuevos reales": sum(d["estado_real"] == NUEVO for d in danos),
            "Nuevos acertados": sum(d["estado_real"] == NUEVO and d["estado_predicho"] == NUEVO for d in danos),
            "Nuevos falsos": (sum(d["estado_real"] != NUEVO and d["estado_predicho"] == NUEVO for d in danos)
                              + sum(f["estado_predicho"] == NUEVO for f in pair["falsos"])),
            "No detectados": sum(d["estado_predicho"] == NO_DETECTADO for d in danos),
            "Exacto": "Sí" if outcome["exacto"] else "No",
            "Error": pair.get("error", ""),
        })
    return rows


def pair_damage_rows(pair: dict) -> List[dict]:
    """Real damages of one pair, then its false detections."""
    rows = []
    for d in pair["danos"]:
        rows.append({"Daño": d["id"], "Tipo real": damage_label(d["tipo"]),
                     "Estado real": state_label(d["estado_real"]), "Estado asignado": state_label(d["estado_predicho"]),
                     "Tipo detectado": damage_label(d["tipo_predicho"]) if d["tipo_predicho"] else "—",
                     "ID A": d["id_a"] or "—", "ID B": d["id_b"] or "—", "IoU A": d["iou_a"], "IoU B": d["iou_b"],
                     "Acierto": "Sí" if d["estado_real"] == d["estado_predicho"] else "No"})
    for f in pair["falsos"]:
        rows.append({"Daño": "(falso)", "Tipo real": "—", "Estado real": "—",
                     "Estado asignado": state_label(f["estado_predicho"]), "Tipo detectado": damage_label(f["tipo"]),
                     "ID A": f["id_a"] or "—", "ID B": f["id_b"] or "—", "IoU A": None, "IoU B": None,
                     "Acierto": "No"})
    return rows


def draw_ground_truth(image_bgr: np.ndarray, pair: dict, side: str = "B") -> np.ndarray:
    """Copy of ``image_bgr`` with the labeled boxes of ``side`` (``"A"`` or ``"B"``) in green.

    The image must be at the labeled resolution: M1 keeps the original size,
    so the change map and the annotated images saved by the evaluation are.
    """
    canvas = image_bgr.copy()
    thickness = max(2, int(round(max(canvas.shape[:2]) / 400)))
    key = "bbox_real_a" if side == "A" else "bbox_real_b"
    for d in pair["danos"]:
        box = d.get(key)
        if box is None:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GROUND_TRUTH_COLOR, thickness)
        label = f"GT {damage_label(d['tipo'])}"
        font_scale = thickness / 3
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = max(0, y1 - h - 2 * thickness)
        cv2.rectangle(canvas, (x1, top), (x1 + w + 2 * thickness, top + h + 2 * thickness), GROUND_TRUTH_COLOR, -1)
        cv2.putText(canvas, label, (x1 + thickness, top + h + thickness), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness)
    return canvas
