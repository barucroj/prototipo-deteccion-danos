"""Bring photos A and B of a pair to the same scale (thesis module M1).

When A and B come at different resolutions (two phones, a re-saved file, an
AI-edited B), the larger one is reduced so that both have the same long side;
the smaller one is **never enlarged**, since that would only add interpolated
pixels. Then a "no change" pair differs only in what the camera saw, not in
how many pixels it has: fine texture (scratches, sensor noise) looks alike in
both and the detectors see the car at the same scale.

The aspect ratio is kept. Pairs taken with the same camera have the same
aspect ratio, so the result is the same size; otherwise both long sides match.
The reduction is geometric: boxes and masks of M3 are in pixels of the reduced
image, and the step (scale, size before and after) goes to the M1 metadata.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.m1_preprocessing.validation import ImageInfo

STEP_NAME = "igualar_tamano_par"

# EXIF orientations 5-8 rotate the stored image by 90 degrees.
_ROTATED_ORIENTATIONS = (5, 6, 7, 8)


def upright_size(info: ImageInfo) -> Tuple[int, int]:
    """``(width, height)`` of the photo after its EXIF orientation is applied."""
    if info.exif_orientation in _ROTATED_ORIENTATIONS:
        return info.height, info.width
    return info.width, info.height


def pair_max_long_sides(size_a: Tuple[int, int],
                        size_b: Tuple[int, int]) -> Tuple[Optional[int], Optional[int]]:
    """Long-side limit for A and for B; ``None`` where the photo stays as it is.

    Only the photo with the larger long side gets a limit, equal to the long
    side of the other one.
    """
    long_a, long_b = max(size_a), max(size_b)
    if long_a > long_b:
        return long_b, None
    if long_b > long_a:
        return None, long_a
    return None, None


def limit_long_side(image_bgr: np.ndarray, max_long_side: Optional[int]) -> Tuple[np.ndarray, Optional[dict]]:
    """Reduce ``image_bgr`` so that its long side is ``max_long_side`` (area interpolation).

    Returns:
        ``(image, step)``; ``step`` is the M1 metadata entry, or ``None`` (and
        the same image) when no reduction is needed.
    """
    if max_long_side is None:
        return image_bgr, None
    if max_long_side < 1:
        raise ValueError(f"max_long_side debe ser >= 1, no {max_long_side}")
    height, width = image_bgr.shape[:2]
    long_side = max(width, height)
    if long_side <= max_long_side:
        return image_bgr, None
    scale = max_long_side / long_side
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    reduced = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
    step = {"nombre": STEP_NAME,
            "parametros": {"escala": round(scale, 6), "de": [int(width), int(height)],
                           "a": [int(new_size[0]), int(new_size[1])]}}
    return reduced, step
