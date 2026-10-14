"""Assign each detected damage to the car part it lies on (thesis module M3).

The damage detector runs on the full image, so it knows *what* a damage is but
not *where on the car* it is. This module answers that by intersecting each
damage mask with the part instance masks from M2:

* a damage belongs to the part instance holding the largest fraction of its
  area, ``area(damage & part) / area(damage)``;
* if no part reaches ``min_overlap`` the damage is left unassigned — it is
  **kept**, never dropped, so a missed part cannot hide a damage;
* ``rel_pos`` is the damage centroid normalized to the assigned part's box,
  which is what M4 compares between the before and after images.

Left/right part pairs are merged into one class (``front_door``), so the class
name does not identify a part. Everything here works on part *instances*,
addressed by their index in the list M2 produced.

Pure numpy on purpose: no torch, no file I/O, so it is testable with
synthetic masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MIN_OVERLAP = 0.5


@dataclass(frozen=True)
class Assignment:
    """Where one damage sits.

    Args:
        part_index: Index of the assigned part instance, or ``None`` when no
            part reached the overlap threshold.
        overlap: Fraction of the damage area inside the best part, in [0, 1].
            Reported even when the damage is unassigned.
        rel_pos: ``(x, y)`` of the damage centroid inside the assigned part's
            box, each in [0, 1]. ``None`` when unassigned.
    """

    part_index: Optional[int]
    overlap: float
    rel_pos: Optional[Tuple[float, float]]


def _crop_window(mask: np.ndarray) -> Optional[Tuple[slice, slice]]:
    """Row/column slices bounding the True pixels, or ``None`` if empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def overlap_fractions(damage_mask: np.ndarray, part_masks: Sequence[np.ndarray]) -> np.ndarray:
    """``area(damage & part) / area(damage)`` for every part.

    Args:
        damage_mask: (H, W) boolean mask of one damage.
        part_masks: (H, W) boolean masks, one per part instance.

    Returns:
        Float array of length ``len(part_masks)``. All zeros if the damage
        mask is empty.
    """
    fractions = np.zeros(len(part_masks), dtype=np.float64)
    window = _crop_window(damage_mask)
    if window is None:
        return fractions

    # Only the damage's bounding window can intersect anything; at 12 MP this
    # avoids a full-image AND per part.
    damage_crop = damage_mask[window]
    area = int(damage_crop.sum())
    for i, part_mask in enumerate(part_masks):
        fractions[i] = np.logical_and(damage_crop, part_mask[window]).sum() / area
    return fractions


def relative_position(damage_mask: np.ndarray, part_box: Sequence[float]) -> Optional[Tuple[float, float]]:
    """Damage centroid normalized to a part box.

    Args:
        damage_mask: (H, W) boolean mask of one damage.
        part_box: ``[x1, y1, x2, y2]`` of the part, in pixels.

    Returns:
        ``(x, y)`` in [0, 1], clipped: a damage that spills past the part box
        still gets a position on its edge. ``None`` if the mask is empty or
        the box has no area.
    """
    x1, y1, x2, y2 = (float(v) for v in part_box)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return None

    ys, xs = np.nonzero(damage_mask)
    if ys.size == 0:
        return None

    rel_x = (xs.mean() - x1) / width
    rel_y = (ys.mean() - y1) / height
    return float(np.clip(rel_x, 0.0, 1.0)), float(np.clip(rel_y, 0.0, 1.0))


def assign_damage(damage_mask: np.ndarray, part_masks: Sequence[np.ndarray],
                  part_boxes: Sequence[Sequence[float]],
                  min_overlap: float = DEFAULT_MIN_OVERLAP) -> Assignment:
    """Assign one damage to the part instance with the largest overlap.

    Args:
        damage_mask: (H, W) boolean mask of the damage.
        part_masks: (H, W) boolean masks, one per part instance.
        part_boxes: ``[x1, y1, x2, y2]`` per part, same order as ``part_masks``.
        min_overlap: Minimum overlap fraction to accept the assignment.

    Ties go to the lowest index. M2 lists parts by descending confidence, so
    that is the more confident part.
    """
    if len(part_masks) != len(part_boxes):
        raise ValueError(
            f"part_masks ({len(part_masks)}) and part_boxes ({len(part_boxes)}) differ in length")

    fractions = overlap_fractions(damage_mask, part_masks)
    if fractions.size == 0:
        return Assignment(part_index=None, overlap=0.0, rel_pos=None)

    best = int(np.argmax(fractions))
    overlap = float(fractions[best])
    if overlap < min_overlap:
        return Assignment(part_index=None, overlap=overlap, rel_pos=None)

    rel_pos = relative_position(damage_mask, part_boxes[best])
    return Assignment(part_index=best, overlap=overlap, rel_pos=rel_pos)


def assign_damages(damage_masks: Sequence[np.ndarray], part_masks: Sequence[np.ndarray],
                   part_boxes: Sequence[Sequence[float]],
                   min_overlap: float = DEFAULT_MIN_OVERLAP) -> List[Assignment]:
    """:func:`assign_damage` for every damage, in order. Never drops one."""
    return [assign_damage(mask, part_masks, part_boxes, min_overlap) for mask in damage_masks]
