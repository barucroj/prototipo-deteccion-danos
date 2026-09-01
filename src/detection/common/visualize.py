"""Drawing helpers for detector predictions and ground truth.

Kept separate from the predict CLI so notebooks (``tests/detection/*.ipynb``)
can render detections the same way the CLI does instead of each growing its
own copy of the drawing code.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

#: Distinct BGR colors cycled per class id, so two adjacent parts or two
#: damage types are visually separable in a single overlay.
_PALETTE = [
    (56, 168, 0), (0, 165, 255), (255, 128, 0), (0, 0, 255), (255, 0, 255),
    (0, 255, 255), (128, 0, 255), (255, 255, 0), (0, 128, 255), (128, 255, 0),
]


def color_for_label(label: int):
    """Stable BGR color for a class id."""
    return _PALETTE[int(label) % len(_PALETTE)]


def draw_detections(image_bgr, boxes, labels, scores, categories,
                    masks=None, mask_alpha: float = 0.45, thickness: int = 2):
    """Draw boxes, class names, scores and (optionally) masks onto an image.

    Args:
        image_bgr: HxWx3 uint8 BGR image; modified in place, also returned.
        boxes: (N, 4) xyxy array.
        labels: (N,) class ids.
        scores: (N,) confidence scores.
        categories: id -> name map, used for the text labels.
        masks: Optional (N, 1, H, W) or (N, H, W) float array of per-instance
            mask probabilities, as Mask R-CNN returns. Values >= 0.5 are
            filled with the class color at ``mask_alpha`` opacity.
        mask_alpha: Opacity of the mask overlay.
        thickness: Box line thickness.
    """
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv-python is required for draw_detections")

    if masks is not None and len(masks):
        masks = np.asarray(masks)
        if masks.ndim == 4:  # (N, 1, H, W) -> (N, H, W)
            masks = masks[:, 0]
        overlay = image_bgr.copy()
        for mask, label in zip(masks, labels):
            overlay[mask >= 0.5] = color_for_label(label)
        cv2.addWeighted(overlay, mask_alpha, image_bgr, 1 - mask_alpha, 0, image_bgr)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = color_for_label(label)
        name = categories.get(int(label), str(int(label)))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image_bgr, f"{name} {score:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image_bgr
