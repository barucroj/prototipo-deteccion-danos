"""Merge duplicate damage detections: same type, same region (thesis module M3).

Mask R-CNN's NMS works on boxes, so one damage can come out twice: two masks
of the same dent that overlap almost entirely, or a long scratch box that holds
a shorter one. In M4 the extra copy becomes a false "new" damage.

Two detections are merged when **all** of these hold:

* same damage type;
* same part instance (``part_index``), or both unassigned — distinct parts
  keep distinct damages, e.g. two broken lamps under one huge detection;
* their masks overlap: IoU >= ``min_iou`` **or** the smaller one lies inside
  the other for at least ``min_containment`` of its area. IoU alone misses a
  short scratch inside a long one (IoU 0.25, containment 0.99 in batch1 p01).

Merging is transitive (connected components). The merged damage has the union
mask, the box of that union and the highest confidence of the group; its part
and ``pos_relativa`` are recomputed on the union mask by the caller.

Thresholds come from three cases of batch1 (p01, p06, p09): **uncalibrated**.

Pure numpy, no torch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MERGE_IOU = 0.5
DEFAULT_MERGE_CONTAINMENT = 0.8


@dataclass(frozen=True)
class MergeSettings:
    """Whether and when same-type detections in the same region are merged."""

    enabled: bool = True
    min_iou: float = DEFAULT_MERGE_IOU
    min_containment: float = DEFAULT_MERGE_CONTAINMENT

    def to_json(self) -> dict:
        record = asdict(self)
        return {"activa": record["enabled"], "iou_minimo": record["min_iou"],
                "contencion_minima": record["min_containment"]}


NO_MERGE = MergeSettings(enabled=False)


def _boxes_touch(box_a: Sequence[float], box_b: Sequence[float]) -> bool:
    return not (box_a[2] <= box_b[0] or box_b[2] <= box_a[0] or box_a[3] <= box_b[1] or box_b[3] <= box_a[1])


def mask_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[float, float]:
    """``(IoU, containment)`` of two boolean masks.

    Containment is ``area(a & b) / min(area(a), area(b))``: 1.0 when the
    smaller mask lies entirely inside the larger one. Both are 0 if a mask is
    empty.
    """
    either = mask_a | mask_b
    rows = np.flatnonzero(either.any(axis=1))
    if rows.size == 0:
        return 0.0, 0.0
    cols = np.flatnonzero(either.any(axis=0))
    window = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
    a, b = mask_a[window], mask_b[window]
    area_a, area_b = int(a.sum()), int(b.sum())
    if area_a == 0 or area_b == 0:
        return 0.0, 0.0
    inter = int(np.logical_and(a, b).sum())
    return inter / (area_a + area_b - inter), inter / min(area_a, area_b)


def merge_groups(names: Sequence[str], boxes: np.ndarray, masks: np.ndarray,
                 part_indices: Sequence[Optional[int]], settings: MergeSettings) -> List[List[int]]:
    """Group the detections that are the same damage.

    Args:
        names: Damage type per detection.
        boxes: (N, 4) xyxy boxes, used only to skip pairs that cannot overlap.
        masks: (N, H, W) boolean masks.
        part_indices: Assigned part instance per detection (``None`` = no part).
        settings: Thresholds; with ``enabled=False`` every detection is its own group.

    Returns:
        Groups of detection indices, each sorted, ordered by their first index
        (so the input order, by descending confidence, is kept). Every index
        appears exactly once.
    """
    n = len(names)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if settings.enabled:
        for i in range(n):
            for j in range(i + 1, n):
                if names[i] != names[j] or part_indices[i] != part_indices[j]:
                    continue
                if not _boxes_touch(boxes[i], boxes[j]):
                    continue
                iou, containment = mask_overlap(masks[i], masks[j])
                if iou >= settings.min_iou or containment >= settings.min_containment:
                    parent[max(root(i), root(j))] = min(root(i), root(j))

    groups: dict = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return [sorted(g) for _, g in sorted(groups.items(), key=lambda item: min(item[1]))]


def mask_box(mask: np.ndarray) -> Optional[List[float]]:
    """xyxy box of the True pixels (exclusive right/bottom edge, like torchvision), ``None`` if empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return [float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)]


def merge_detections(names: Sequence[str], scores: np.ndarray, boxes: np.ndarray, masks: np.ndarray,
                     groups: Sequence[Sequence[int]]):
    """Collapse each group into one detection.

    Returns:
        ``(names, scores, boxes, masks)`` with one entry per group: the
        group's type, its highest score, the box of the union mask (the
        union of the boxes if the masks are empty) and the union mask.
    """
    out_names, out_scores, out_boxes, out_masks = [], [], [], []
    for group in groups:
        group = list(group)
        out_names.append(names[group[0]])
        out_scores.append(float(np.max(scores[group])))
        if len(group) == 1:
            out_boxes.append(np.asarray(boxes[group[0]], dtype=float))
            out_masks.append(masks[group[0]])
            continue
        union = np.logical_or.reduce(masks[group], axis=0)
        box = mask_box(union)
        if box is None:
            members = np.asarray(boxes[group], dtype=float)
            box = [members[:, 0].min(), members[:, 1].min(), members[:, 2].max(), members[:, 3].max()]
        out_boxes.append(np.asarray(box, dtype=float))
        out_masks.append(union)

    height, width = masks.shape[1:] if masks.ndim == 3 else (0, 0)
    return (out_names, np.asarray(out_scores, dtype=float), np.asarray(out_boxes, dtype=float).reshape(-1, 4),
            np.asarray(out_masks, dtype=bool).reshape(-1, height, width))


def add_merge_arguments(parser) -> None:
    """``--no-merge``, ``--merge-iou`` and ``--merge-containment`` for the CLIs."""
    parser.add_argument("--no-merge", action="store_true",
                        help="do not merge same-type damages in the same region (M3)")
    parser.add_argument("--merge-iou", type=float, default=DEFAULT_MERGE_IOU,
                        help="mask IoU from which two same-type damages are merged")
    parser.add_argument("--merge-containment", type=float, default=DEFAULT_MERGE_CONTAINMENT,
                        help="fraction of the smaller mask inside the other from which they are merged")


def merge_settings_from_args(args) -> MergeSettings:
    return MergeSettings(enabled=not args.no_merge, min_iou=args.merge_iou,
                         min_containment=args.merge_containment)
