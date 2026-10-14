"""Run M2 and M3 on one image and build the M3 report (thesis module M3).

Glue between the two detectors and :mod:`src.m3_damage.assignment.assign`:

1. the parts Mask R-CNN gives the part instances of the image;
2. the damage Mask R-CNN gives the damages, on the **full image** (no crops);
3. each damage is assigned to a part instance and gets its ``pos_relativa``.

Both models are passed in already loaded, so a caller (the orchestrator, the
manual demo) loads each one once and reuses it for image A and image B.

:func:`build_report` is torch-free and takes plain arrays, so the report
format is tested with synthetic detections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.m3_damage.assignment.assign import DEFAULT_MIN_OVERLAP, Assignment, assign_damages

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
    """

    report: dict
    parts: Detections
    damages: Detections
    assignments: List[Assignment]


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
                 min_overlap: float = DEFAULT_MIN_OVERLAP) -> dict:
    """Build the M3 JSON for one image.

    Args:
        parts / damages: Detections of each model on the image.
        assignments: One :class:`Assignment` per damage, same order.
        image_label: Which image of the pair this is (``"A"`` or ``"B"``);
            prefixes the damage ids so M4 can mix both images' damages.
        width / height: Image size in pixels.
        file_name: Source file name, for traceability.

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

    danos = []
    for i, assignment in enumerate(assignments):
        assigned = assignment.part_index is not None
        danos.append({
            "id": f"{image_label}-{i + 1:02d}",
            "tipo": damages.names[i],
            "confianza": round(float(damages.scores[i]), 4),
            "bbox": _round_box(damages.boxes[i]),
            "area_px": int(damages.masks[i].sum()),
            "parte": parts.names[assignment.part_index] if assigned else None,
            "parte_id": part_ids[assignment.part_index] if assigned else None,
            "solapamiento": round(assignment.overlap, 4),
            "pos_relativa": [round(v, 4) for v in assignment.rel_pos] if assignment.rel_pos else None,
        })

    return {
        "imagen": image_label,
        "archivo": file_name,
        "width": int(width),
        "height": int(height),
        "umbral_partes": part_threshold,
        "umbral_danos": damage_threshold,
        "umbral_solapamiento": min_overlap,
        "partes": partes,
        "danos": danos,
    }


def analyze_image(image_bgr: np.ndarray, parts_model, part_categories: Dict[int, str],
                  damage_model, damage_categories: Dict[int, str], device,
                  image_label: str = "A", file_name: Optional[str] = None,
                  part_threshold: float = DEFAULT_PART_THRESHOLD,
                  damage_threshold: float = DEFAULT_DAMAGE_THRESHOLD,
                  min_overlap: float = DEFAULT_MIN_OVERLAP) -> ImageAnalysis:
    """Detect parts and damages on one image and assign each damage to a part."""
    height, width = image_bgr.shape[:2]
    parts = detect(parts_model, part_categories, image_bgr, device, part_threshold)
    damages = detect(damage_model, damage_categories, image_bgr, device, damage_threshold)
    assignments = assign_damages(damages.masks, parts.masks, parts.boxes, min_overlap)
    report = build_report(parts, damages, assignments, image_label, width, height,
                          file_name=file_name, part_threshold=part_threshold,
                          damage_threshold=damage_threshold, min_overlap=min_overlap)
    return ImageAnalysis(report=report, parts=parts, damages=damages, assignments=assignments)
