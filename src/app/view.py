"""Turn M3 and M4 reports into what the web interface shows. No Streamlit, no torch.

Kept apart from :mod:`src.app.streamlit_app` so the tables and counts are
tested with plain dicts.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

#: CarDD class name -> label shown to the user.
DAMAGE_LABELS = {
    "dent": "Golpe",
    "scratch": "Rayón",
    "crack": "Fisura",
    "glass shatter": "Cristal roto",
    "lamp broken": "Faro roto",
    "tire flat": "Llanta ponchada",
}
#: M4 state -> label shown to the user.
STATE_LABELS = {"nuevo": "Nuevo", "preexistente": "Preexistente", "no_visible": "No visible"}
UNASSIGNED = "Sin parte asignada"


def damage_label(tipo: str) -> str:
    return DAMAGE_LABELS.get(tipo, tipo)


def damage_rows(report: dict) -> List[dict]:
    """One row per damage of an M3 report, for a table."""
    rows = []
    for dano in report["danos"]:
        rows.append({
            "ID": dano["id"],
            "Tipo": damage_label(dano["tipo"]),
            "Confianza": round(dano["confianza"], 2),
            "Parte": f"{dano['parte']} ({dano['parte_id']})" if dano["parte"] else UNASSIGNED,
            "Solapamiento": round(dano["solapamiento"], 2),
            "Área (px)": dano["area_px"],
        })
    return rows


def part_rows(report: dict) -> List[dict]:
    """One row per part instance of an M3 report."""
    return [{"ID": p["id"], "Parte": p["clase_parte"], "Confianza": round(p["confianza"], 2),
             "Área (px)": p["area_px"]} for p in report["partes"]]


def comparison_rows(report: dict) -> List[dict]:
    """One row per damage of an M4 report, the new ones first."""
    order = list(STATE_LABELS)
    rows = []
    for dano in sorted(report["danos"], key=lambda d: order.index(d["estado"])):
        rows.append({
            "Estado": STATE_LABELS[dano["estado"]],
            "Tipo": damage_label(dano["tipo"]),
            "Parte": dano["parte"] or UNASSIGNED,
            "Confianza": round(dano["confianza"], 2),
            "ID A": dano["id_a"] or "—",
            "ID B": dano["id_b"] or "—",
            "Distancia": dano["distancia"],
            "Revisar": "Sí" if dano["requiere_revision"] else "No",
        })
    return rows


def to_display(image_bgr: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """RGB copy no larger than ``max_side``: a 12 MP photo is slow to send to the browser."""
    height, width = image_bgr.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1:
        image_bgr = cv2.resize(image_bgr, (int(round(width * scale)), int(round(height * scale))),
                               interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
