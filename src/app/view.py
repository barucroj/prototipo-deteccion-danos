"""Turn M3 reports into what the web interface shows. No Streamlit, no torch.

Kept apart from :mod:`src.app.streamlit_app` so the tables and counts are
tested with plain dicts.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

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


def count_by_type(report: dict) -> Dict[str, int]:
    return dict(Counter(damage_label(d["tipo"]) for d in report["danos"]))


def type_count_rows(report_a: dict, report_b: dict) -> List[dict]:
    """Damage counts per type in A and in B, every type present in either.

    Descriptive only: a type with the same count in both images may still
    hold new damages. Telling new from pre-existing is M4's job.
    """
    counts_a, counts_b = count_by_type(report_a), count_by_type(report_b)
    order = [label for label in DAMAGE_LABELS.values() if label in counts_a or label in counts_b]
    order += sorted((set(counts_a) | set(counts_b)) - set(order))
    return [{"Tipo": t, "A (antes)": counts_a.get(t, 0), "B (después)": counts_b.get(t, 0)} for t in order]


def to_display(image_bgr: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """RGB copy no larger than ``max_side``: a 12 MP photo is slow to send to the browser."""
    height, width = image_bgr.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1:
        image_bgr = cv2.resize(image_bgr, (int(round(width * scale)), int(round(height * scale))),
                               interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
