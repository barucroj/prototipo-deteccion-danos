"""Write an M3 result to disk: JSON, one mask PNG per damage, annotated image.

Layout under ``output_dir`` for an image named ``car.jpg``::

    car.json            the M3 report
    car_anotada.jpg     damages and their parts drawn on the image
    masks/car_A-01.png  one binary mask (0/255) per damage

Mask paths in the JSON are relative to ``output_dir``, so the folder can be
moved as a unit.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import cv2
import numpy as np

from src.m3_damage.assignment.analyze import ImageAnalysis
from src.m3_damage.assignment.render import draw_analysis

MASKS_SUBDIR = "masks"


def save_analysis(analysis: ImageAnalysis, image_bgr: np.ndarray, output_dir: str, stem: str) -> Dict[str, str]:
    """Write the report, the damage masks and the annotated image.

    Adds a ``mascara`` path to every damage of ``analysis.report`` before
    writing it.

    Args:
        analysis: Result of :func:`src.m3_damage.assignment.analyze.analyze_image`.
        image_bgr: The image that was analyzed.
        output_dir: Destination folder, created if missing.
        stem: Base name for the files (usually the image name without extension).

    Returns:
        ``{"json": ..., "annotated": ...}`` with the paths written.
    """
    os.makedirs(os.path.join(output_dir, MASKS_SUBDIR), exist_ok=True)

    for entry, mask in zip(analysis.report["danos"], analysis.damages.masks):
        relative = f"{MASKS_SUBDIR}/{stem}_{entry['id']}.png"
        cv2.imwrite(os.path.join(output_dir, relative), mask.astype(np.uint8) * 255)
        entry["mascara"] = relative

    json_path = os.path.join(output_dir, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(analysis.report, handle, indent=2, ensure_ascii=False)

    annotated_path = os.path.join(output_dir, f"{stem}_anotada.jpg")
    cv2.imwrite(annotated_path, draw_analysis(image_bgr, analysis))
    return {"json": json_path, "annotated": annotated_path}
