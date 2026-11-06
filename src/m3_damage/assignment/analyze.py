"""Run M2 and M3 on one image and build the M3 report (thesis module M3).

Glue between the two detectors and :mod:`src.m3_damage.assignment.assign`:

1. the parts Mask R-CNN gives the part instances of the image;
2. the damage Mask R-CNN gives the damages, on the **full image** (no crops);
3. each damage is assigned to a part instance and gets its ``pos_relativa``;
4. same-type detections of one damage in the same part are merged
   (:mod:`src.m3_damage.assignment.merge`) and the merged ones reassigned.

Both models are passed in already loaded, so a caller (the orchestrator, the
manual demo) loads each one once and reuses it for image A and image B.

:func:`build_report` is torch-free and takes plain arrays, so the report
format is tested with synthetic detections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.m3_damage.assignment.assign import DEFAULT_MIN_OVERLAP, Assignment, assign_damage, assign_damages
from src.m3_damage.assignment.merge import NO_MERGE, MergeSettings, merge_detections, merge_groups

#: Minimum confidence for a part instance to take part in the assignment.
DEFAULT_PART_THRESHOLD = 0.7
#: Minimum confidence for a damage to be reported.
DEFAULT_DAMAGE_THRESHOLD = 0.5


@dataclass
class Detections:
    """One detector's output on one image, filtered by confidence.

    Args:
        names: Class name per instance.
        scores: (N,) confidences, descending (torchvision's order).
        boxes: (N, 4) xyxy boxes in pixels.
        masks: (N, H, W) boolean instance masks.
    """

    names: List[str]
    scores: np.ndarray
    boxes: np.ndarray
    masks: np.ndarray

    def __len__(self) -> int:
        return len(self.names)


@dataclass
class ImageAnalysis:
    """Everything M3 knows about one image.

    ``report`` is the JSON-serializable result; the masks stay alongside it
    for drawing and for writing one PNG per damage.

    ``damages`` and ``assignments`` are after merging, aligned with
    ``report["danos"]``. ``raw_damages`` / ``raw_assignments`` keep the
    detector's output before merging (``None`` = same as the merged ones), so
    a higher confidence threshold can be applied and merged again
    (:func:`restrict_damages`, :func:`split_support`).
    """

    report: dict
    parts: Detections
    damages: Detections
    assignments: List[Assignment]
    raw_damages: Optional[Detections] = None
    raw_assignments: Optional[List[Assignment]] = None
    merge: MergeSettings = NO_MERGE


def detect(model, categories: Dict[int, str], image_bgr: np.ndarray, device,
           score_threshold: float) -> Detections:
    """Run one Mask R-CNN on a BGR image and keep instances above the threshold."""
    import cv2
    import torch
    from torchvision.transforms import functional as F

    tensor = F.to_tensor(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).to(device)
    with torch.no_grad():
        output = model([tensor])[0]

    keep = output["scores"] >= score_threshold
    labels = output["labels"][keep].cpu().numpy()
    # Binarize on the device: float masks at 12 MP are 48 MB per instance on
    # the CPU, boolean ones 12 MB.
    masks = (output["masks"][keep, 0] >= 0.5).cpu().numpy()
    return Detections(
        names=[categories.get(int(label), str(int(label))) for label in labels],
        scores=output["scores"][keep].cpu().numpy(),
        boxes=output["boxes"][keep].cpu().numpy(),
        masks=masks,
    )


def _round_box(box: Sequence[float]) -> List[float]:
    return [round(float(v), 1) for v in box]


def build_report(parts: Detections, damages: Detections, assignments: Sequence[Assignment],
                 image_label: str, width: int, height: int,
                 file_name: Optional[str] = None,
                 part_threshold: float = DEFAULT_PART_THRESHOLD,
                 damage_threshold: float = DEFAULT_DAMAGE_THRESHOLD,
                 min_overlap: float = DEFAULT_MIN_OVERLAP,
                 damage_ids: Optional[Sequence[str]] = None,
                 merged_from: Optional[Sequence[Optional[List[str]]]] = None,
                 merge: Optional[MergeSettings] = None) -> dict:
    """Build the M3 JSON for one image.

    Args:
        parts / damages: Detections of each model on the image.
        assignments: One :class:`Assignment` per damage, same order.
        image_label: Which image of the pair this is (``"A"`` or ``"B"``);
            prefixes the damage ids so M4 can mix both images' damages.
        width / height: Image size in pixels.
        file_name: Source file name, for traceability.
        damage_ids: Id per damage; default ``<label>-01``, ``<label>-02``...
        merged_from: Per damage, the ids of the detections merged into it
            (``fusionados``), ``None`` for a damage that was not merged.
        merge: Merge settings, recorded as ``fusion`` (``None`` = not recorded).

    A damage with no part keeps ``parte``, ``parte_id`` and ``pos_relativa``
    as ``None``; ``solapamiento`` still reports its best overlap.
    ``parte_id`` identifies the part *instance*: left/right pairs share a
    class name, so ``parte`` alone does not say which door it is.
    """
    if len(assignments) != len(damages):
        raise ValueError(f"{len(damages)} damages but {len(assignments)} assignments")

    part_ids = [f"p-{i + 1:02d}" for i in range(len(parts))]
    partes = [
        {
            "id": part_ids[i],
            "clase_parte": parts.names[i],
            "confianza": round(float(parts.scores[i]), 4),
            "bbox": _round_box(parts.boxes[i]),
            "area_px": int(parts.masks[i].sum()),
        }
        for i in range(len(parts))
    ]

    if damage_ids is None:
        damage_ids = [f"{image_label}-{i + 1:02d}" for i in range(len(damages))]
    danos = []
    for i, assignment in enumerate(assignments):
        assigned = assignment.part_index is not None
        danos.append({
            "id": damage_ids[i],
            "tipo": damages.names[i],
            "confianza": round(float(damages.scores[i]), 4),
            "bbox": _round_box(damages.boxes[i]),
            "area_px": int(damages.masks[i].sum()),
            "parte": parts.names[assignment.part_index] if assigned else None,
            "parte_id": part_ids[assignment.part_index] if assigned else None,
            "solapamiento": round(assignment.overlap, 4),
            "pos_relativa": [round(v, 4) for v in assignment.rel_pos] if assignment.rel_pos else None,
            "fusionados": list(merged_from[i]) if merged_from is not None and merged_from[i] else None,
        })

    return {
        "imagen": image_label,
        "archivo": file_name,
        "width": int(width),
        "height": int(height),
        "umbral_partes": part_threshold,
        "umbral_danos": damage_threshold,
        "umbral_solapamiento": min_overlap,
        "fusion": merge.to_json() if merge is not None else None,
        "partes": partes,
        "danos": danos,
    }


def finish_analysis(parts: Detections, damages: Detections, assignments: Sequence[Assignment],
                    image_label: str, width: int, height: int, file_name: Optional[str] = None,
                    part_threshold: float = DEFAULT_PART_THRESHOLD,
                    damage_threshold: float = DEFAULT_DAMAGE_THRESHOLD,
                    min_overlap: float = DEFAULT_MIN_OVERLAP,
                    merge: MergeSettings = MergeSettings()) -> ImageAnalysis:
    """Merge duplicate damages, reassign the merged ones and build the report.

    ``damages`` / ``assignments`` are the detector's output already assigned
    to parts. Ids are given before merging (``B-01``, ``B-02``... by
    descending confidence); a merged damage keeps the id of its most
    confident member and lists every member in ``fusionados``, so ids can
    have gaps.
    """
    assignments = list(assignments)
    raw_ids = [f"{image_label}-{i + 1:02d}" for i in range(len(damages))]
    groups = merge_groups(damages.names, damages.boxes, damages.masks,
                          [a.part_index for a in assignments], merge)
    if all(len(g) == 1 for g in groups):
        merged = damages  # nothing to merge: no copy of the (12 MB each at 12 MP) masks
    else:
        names, scores, boxes, masks = merge_detections(damages.names, damages.scores, damages.boxes,
                                                       damages.masks, groups)
        merged = Detections(names=names, scores=scores, boxes=boxes, masks=masks)
    masks = merged.masks
    merged_assignments = [
        assignments[g[0]] if len(g) == 1 else assign_damage(masks[k], parts.masks, parts.boxes, min_overlap)
        for k, g in enumerate(groups)
    ]
    report = build_report(parts, merged, merged_assignments, image_label, width, height, file_name=file_name,
                          part_threshold=part_threshold, damage_threshold=damage_threshold,
                          min_overlap=min_overlap, damage_ids=[raw_ids[g[0]] for g in groups],
                          merged_from=[[raw_ids[i] for i in g] if len(g) > 1 else None for g in groups],
                          merge=merge)
    return ImageAnalysis(report=report, parts=parts, damages=merged, assignments=merged_assignments,
                         raw_damages=damages, raw_assignments=assignments, merge=merge)


def analyze_image(image_bgr: np.ndarray, parts_model, part_categories: Dict[int, str],
                  damage_model, damage_categories: Dict[int, str], device,
                  image_label: str = "A", file_name: Optional[str] = None,
                  part_threshold: float = DEFAULT_PART_THRESHOLD,
                  damage_threshold: float = DEFAULT_DAMAGE_THRESHOLD,
                  min_overlap: float = DEFAULT_MIN_OVERLAP,
                  merge: MergeSettings = MergeSettings()) -> ImageAnalysis:
    """Detect parts and damages on one image, assign each damage to a part and merge duplicates."""
    height, width = image_bgr.shape[:2]
    parts = detect(parts_model, part_categories, image_bgr, device, part_threshold)
    damages = detect(damage_model, damage_categories, image_bgr, device, damage_threshold)
    assignments = assign_damages(damages.masks, parts.masks, parts.boxes, min_overlap)
    return finish_analysis(parts, damages, assignments, image_label, width, height, file_name=file_name,
                           part_threshold=part_threshold, damage_threshold=damage_threshold,
                           min_overlap=min_overlap, merge=merge)


def restrict_damages(analysis: ImageAnalysis, threshold: float) -> ImageAnalysis:
    """The M3 result as if the damage detector had run at ``threshold`` (>= the original one).

    Filters the detections **before** merging and merges again with the same
    settings: a low-confidence duplicate that the threshold drops must not
    keep enlarging the merged damage. The detector returns them by descending
    confidence, so a detection keeps its id at every threshold.
    """
    raw = analysis.raw_damages if analysis.raw_damages is not None else analysis.damages
    raw_assignments = analysis.raw_assignments if analysis.raw_assignments is not None else analysis.assignments
    keep = np.flatnonzero(raw.scores >= threshold)
    damages = Detections(names=[raw.names[i] for i in keep], scores=raw.scores[keep],
                         boxes=raw.boxes[keep], masks=raw.masks[keep])
    assignments = [raw_assignments[i] for i in keep]
    old = analysis.report
    return finish_analysis(analysis.parts, damages, assignments, old["imagen"], old["width"], old["height"],
                           file_name=old["archivo"], part_threshold=old["umbral_partes"],
                           damage_threshold=threshold, min_overlap=old["umbral_solapamiento"],
                           merge=analysis.merge)


def split_support(analysis: ImageAnalysis, threshold: float) -> Tuple[ImageAnalysis, ImageAnalysis]:
    """Split a low-threshold analysis into the damages and the hysteresis support.

    Args:
        analysis: M3 run at the low (support) threshold.
        threshold: The damage threshold.

    Returns:
        ``(main, support)``: ``main`` is the analysis at ``threshold``;
        ``support`` keeps, from the low-threshold analysis, only the merged
        damages whose confidence is below ``threshold`` (a group with any
        member above it is already a damage of ``main``). Their ids never
        collide with the ones of ``main``.
    """
    main = restrict_damages(analysis, threshold)
    keep = [i for i, score in enumerate(analysis.damages.scores) if score < threshold]
    damages = analysis.damages
    support_damages = Detections(names=[damages.names[i] for i in keep], scores=damages.scores[keep],
                                 boxes=damages.boxes[keep], masks=damages.masks[keep])
    report = dict(analysis.report, danos=[analysis.report["danos"][i] for i in keep])
    support = ImageAnalysis(report=report, parts=analysis.parts, damages=support_damages,
                            assignments=[analysis.assignments[i] for i in keep], merge=analysis.merge)
    return main, support
