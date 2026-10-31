"""Build and write the M4 report of one A/B pair (thesis module M4).

The report follows the data model of the thesis (``ReporteComparacion`` and
``DanoComparado``, section 4.9.2): the state of every damage plus what RF09
asks of each one: type, part, location and confidence. Works on M3 JSON
reports, so it is testable without models.

A damage of A with no pair has no pixels in B. ``bbox_en_b`` is where it would
be expected there: through the matched part when there is one, otherwise by
scaling to the size of B. It is an estimate, used to draw the change map.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Mapping, Optional

import cv2
import numpy as np

from src.m4_comparison.damages import (DEFAULT_DAMAGE_MAX_DISTANCE, NO_VISIBLE, NUEVO, PREEXISTENTE,
                                       DamageStatus, compare_reports)
from src.m4_comparison.parts import DEFAULT_PART_MAX_DISTANCE, PartMatching

REPORT_FILE = "reporte_par.json"
CHANGE_MAP_FILE = "mapa_cambios.png"


def _scale_box(bbox, report_a: dict, report_b: dict) -> List[float]:
    scale_x, scale_y = report_b["width"] / report_a["width"], report_b["height"] / report_a["height"]
    x1, y1, x2, y2 = bbox
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def _through_part(bbox, part_a: dict, part_b: dict) -> List[float]:
    """Move a box from a part of A to the same relative place of its pair in B."""
    ax1, ay1, ax2, ay2 = part_a["bbox"]
    bx1, by1, bx2, by2 = part_b["bbox"]
    scale_x = (bx2 - bx1) / max(ax2 - ax1, 1e-6)
    scale_y = (by2 - by1) / max(ay2 - ay1, 1e-6)
    x1, y1, x2, y2 = bbox
    return [bx1 + (x1 - ax1) * scale_x, by1 + (y1 - ay1) * scale_y,
            bx1 + (x2 - ax1) * scale_x, by1 + (y2 - ay1) * scale_y]


def expected_box_in_b(dano_a: dict, report_a: dict, report_b: dict, partes: PartMatching) -> List[float]:
    """Where a damage of A should appear in B, clipped to the image and rounded."""
    part_b_id = partes.a_to_b().get(dano_a.get("parte_id"))
    if part_b_id is None:
        box = _scale_box(dano_a["bbox"], report_a, report_b)
    else:
        part_a = next(p for p in report_a["partes"] if p["id"] == dano_a["parte_id"])
        part_b = next(p for p in report_b["partes"] if p["id"] == part_b_id)
        box = _through_part(dano_a["bbox"], part_a, part_b)
    limits = (report_b["width"], report_b["height"]) * 2
    return [round(float(np.clip(value, 0, limit)), 1) for value, limit in zip(box, limits)]


def _compared_damage(status: DamageStatus, by_id_a: Mapping[str, dict], by_id_b: Mapping[str, dict],
                     report_a: dict, report_b: dict, partes: PartMatching) -> dict:
    # The damage in B describes the pair; a not visible one exists only in A.
    dano = by_id_b[status.id_b] if status.id_b else by_id_a[status.id_a]
    if status.estado == NO_VISIBLE:
        box_in_b = expected_box_in_b(dano, report_a, report_b, partes)
    else:
        box_in_b = dano["bbox"]
    return {
        "estado": status.estado,
        "tipo": status.tipo,
        "parte": status.parte,
        "parte_id": dano["parte_id"],
        "id_a": status.id_a,
        "id_b": status.id_b,
        "confianza": dano["confianza"],
        "bbox": dano["bbox"],
        "bbox_en_b": box_in_b,
        "pos_relativa": dano["pos_relativa"],
        "distancia": status.distancia,
        "metodo": status.metodo,
        "requiere_revision": status.estado == NO_VISIBLE or status.parte is None,
    }


def build_report(report_a: dict, report_b: dict,
                 part_max_distance: float = DEFAULT_PART_MAX_DISTANCE,
                 damage_max_distance: float = DEFAULT_DAMAGE_MAX_DISTANCE,
                 modelos: Optional[Dict[str, str]] = None) -> dict:
    """Compare two M3 reports and return the M4 report of the pair.

    Args:
        report_a / report_b: M3 reports of the before and after images.
        part_max_distance / damage_max_distance: Thresholds of M4, recorded in
            the report.
        modelos: Checkpoints that produced the M3 reports, ``None`` if unknown.
    """
    comparison = compare_reports(report_a, report_b, part_max_distance, damage_max_distance)
    by_id_a = {d["id"]: d for d in report_a["danos"]}
    by_id_b = {d["id"]: d for d in report_b["danos"]}
    return {
        "imagen_a": report_a.get("archivo"),
        "imagen_b": report_b.get("archivo"),
        "umbral_distancia": damage_max_distance,
        "umbral_distancia_partes": part_max_distance,
        "modelos": modelos,
        "resumen": {
            "nuevos": comparison.count(NUEVO),
            "preexistentes": comparison.count(PREEXISTENTE),
            "no_visibles": comparison.count(NO_VISIBLE),
        },
        "danos": [_compared_damage(status, by_id_a, by_id_b, report_a, report_b, comparison.partes)
                  for status in comparison.danos],
        "mapa_cambios": CHANGE_MAP_FILE,
    }


def save_report(report: dict, change_map: np.ndarray, output_dir: str) -> Dict[str, str]:
    """Write the JSON and the change map; ``mapa_cambios`` is relative to the JSON.

    Returns:
        ``{"json": ..., "change_map": ...}`` with the paths written.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, REPORT_FILE)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    map_path = os.path.join(output_dir, report["mapa_cambios"])
    cv2.imwrite(map_path, change_map)
    return {"json": json_path, "change_map": map_path}
