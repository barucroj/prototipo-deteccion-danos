"""Draw an M3 result: the damages and the part each one sits on.

Unlike :func:`src.detection_common.visualize.draw_detections`, which draws
every instance of one detector, this shows only what the report is about:
each damage filled in, and the outline of the part it was assigned to. Parts
without a damage are not drawn.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.m3_damage.assignment.analyze import ImageAnalysis
from src.detection_common.visualize import color_for_label, put_label

#: BGR outline color of a part that holds a damage.
PART_COLOR = (255, 255, 255)
UNASSIGNED_LABEL = "sin parte asignada"
NO_DAMAGE_LABEL = "Sin danos detectados"


def draw_analysis(image_bgr: np.ndarray, analysis: ImageAnalysis, mask_alpha: float = 0.5) -> np.ndarray:
    """Return a copy of ``image_bgr`` with the damages and their parts drawn.

    Line widths and text scale follow the image size, so a 4000x3000 photo
    and a 640x640 one are equally readable once fitted to the screen.
    """
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    scale = max(0.5, max(width, height) / 1600)
    line = max(2, int(round(scale * 2)))
    damages, parts = analysis.damages, analysis.parts

    if len(damages) == 0:
        put_label(canvas, NO_DAMAGE_LABEL, (int(10 * scale), int(40 * scale)), scale * 1.2, PART_COLOR)
        return canvas

    # Part outlines first, so the damage fill and labels stay on top.
    used_parts = sorted({a.part_index for a in analysis.assignments if a.part_index is not None})
    for index in used_parts:
        contours, _ = cv2.findContours(parts.masks[index].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, PART_COLOR, line)

    overlay = canvas.copy()
    for i in range(len(damages)):
        overlay[damages.masks[i]] = color_for_label(i)
    cv2.addWeighted(overlay, mask_alpha, canvas, 1 - mask_alpha, 0, canvas)

    for index in used_parts:
        x1, y1 = parts.boxes[index][:2]
        part_id = analysis.report["partes"][index]["id"]
        put_label(canvas, f"{parts.names[index]} ({part_id})", (x1, y1 + 30 * scale), scale * 0.8, PART_COLOR)

    for i, entry in enumerate(analysis.report["danos"]):
        x1, y1, x2, y2 = (int(v) for v in damages.boxes[i])
        color = color_for_label(i)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line)
        where = entry["parte"] or UNASSIGNED_LABEL
        put_label(canvas, f"{entry['id']} {entry['tipo']} {entry['confianza']:.2f} -> {where}",
                   (x1, y1 - 6 * scale), scale * 0.8, color)
    return canvas
