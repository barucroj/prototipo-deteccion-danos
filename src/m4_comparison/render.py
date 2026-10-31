"""Draw the change map of a pair on image B (thesis module M4).

State is shown by shape as well as color, so the map still reads in grayscale:
a new damage has a strong fill and a thick box, a preexisting one a faint fill
and a thin box, and a not visible one only a dashed box at its expected place,
since it has no pixels in B. A forced comparison carries a warning on the map.
"""

from __future__ import annotations

from typing import Mapping, Optional

import cv2
import numpy as np

from src.detection_common.visualize import put_label
from src.m4_comparison.damages import NO_VISIBLE, NUEVO, PREEXISTENTE

#: BGR color per state.
STATE_COLORS = {NUEVO: (40, 40, 230), PREEXISTENTE: (0, 190, 255), NO_VISIBLE: (230, 150, 40)}
STATE_TEXT = {NUEVO: "NUEVO", PREEXISTENTE: "PREEXISTENTE", NO_VISIBLE: "NO VISIBLE"}
SUMMARY_KEYS = {NUEVO: "nuevos", PREEXISTENTE: "preexistentes", NO_VISIBLE: "no_visibles"}
FILL_ALPHA = {NUEVO: 0.55, PREEXISTENTE: 0.25}
FORCED_TEXT = "PAR NO VERIFICADO: REQUIERE REVISION"


def _dashed_box(image_bgr, box, color, line: int) -> None:
    x1, y1, x2, y2 = box
    dash = max(6, line * 4)
    for x in range(x1, x2, dash * 2):
        for y in (y1, y2):
            cv2.line(image_bgr, (x, y), (min(x + dash, x2), y), color, line)
    for y in range(y1, y2, dash * 2):
        for x in (x1, x2):
            cv2.line(image_bgr, (x, y), (x, min(y + dash, y2)), color, line)


def _fill(canvas, mask, color, alpha: float) -> None:
    canvas[mask] = (canvas[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def draw_change_map(image_b: np.ndarray, report: dict,
                    masks_b: Optional[Mapping[str, np.ndarray]] = None) -> np.ndarray:
    """Return a copy of ``image_b`` with every damage of the M4 report drawn.

    Args:
        image_b: The BGR image B.
        report: M4 report from :func:`src.m4_comparison.report.build_report`.
        masks_b: Boolean mask per damage id of B. A damage without one gets
            only its box.
    """
    canvas = image_b.copy()
    height, width = canvas.shape[:2]
    scale = max(0.5, max(width, height) / 1600)
    line = max(2, int(round(scale * 2)))
    masks_b = masks_b or {}

    # New damages last, so they stay on top of everything else.
    order = sorted(report["danos"], key=lambda d: d["estado"] == NUEVO)
    for dano in order:
        mask = masks_b.get(dano["id_b"])
        if mask is not None and dano["estado"] in FILL_ALPHA:
            _fill(canvas, mask, STATE_COLORS[dano["estado"]], FILL_ALPHA[dano["estado"]])

    for dano in order:
        estado = dano["estado"]
        color = STATE_COLORS[estado]
        x1, y1, x2, y2 = (int(v) for v in dano["bbox_en_b"])
        if estado == NO_VISIBLE:
            _dashed_box(canvas, (x1, y1, x2, y2), color, line)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line * 2 if estado == NUEVO else line)
        damage_id = dano["id_b"] or dano["id_a"]
        put_label(canvas, f"{damage_id} {STATE_TEXT[estado]} {dano['tipo']} {dano['confianza']:.2f}",
                  (x1, y1 - 6 * scale), scale * 0.7, color)

    for row, (estado, key) in enumerate(SUMMARY_KEYS.items()):
        put_label(canvas, f"{STATE_TEXT[estado]}: {report['resumen'][key]}",
                  (10 * scale, (34 + row * 34) * scale), scale * 0.8, STATE_COLORS[estado])
    if report.get("comparacion_forzada"):
        put_label(canvas, FORCED_TEXT, (10 * scale, (34 + len(SUMMARY_KEYS) * 34) * scale), scale * 0.8,
                  (0, 255, 255))
    return canvas
