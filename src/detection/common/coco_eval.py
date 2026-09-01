"""Standard COCO evaluation via ``pycocotools``.

This is the metric to quote in the thesis: it is the same AP definition used
by the COCO benchmark and by every paper that reports on CarDD, so numbers
produced here are comparable to published ones. The from-scratch mAP@0.5 in
:mod:`src.detection.common.engine` remains as a fallback for environments
without ``pycocotools`` — its numbers are NOT comparable to these (different
matching rules, ``maxDets``, and area ranges).

Supports both ``bbox`` and ``segm`` IoU types; the damage detector is scored
on both, since mask AP is the honest measure of how well a thin diagonal
scratch is localized.
"""

from __future__ import annotations

import contextlib
import io as _io

import numpy as np
import torch
from tqdm import tqdm

try:
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    PYCOCOTOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYCOCOTOOLS_AVAILABLE = False

#: Index of each summary statistic in ``COCOeval.stats``.
STAT_NAMES = (
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
)


def _encode_mask(mask: np.ndarray) -> dict:
    """RLE-encode a binary HxW mask for COCOeval's ``segm`` IoU type."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


@torch.no_grad()
def collect_detections(model, data_loader, device, with_masks: bool = False,
                       score_threshold: float = 0.0):
    """Run the model over a loader and return COCO-format detection dicts.

    Args:
        score_threshold: Detections below this score are dropped. Keep at 0
            for evaluation — AP integrates over the whole precision/recall
            curve, so filtering low-score detections only removes recall the
            metric would have credited.
    """
    model.eval()
    detections = []

    pbar = tqdm(data_loader, desc="Evaluating", leave=True, unit="batch")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"])
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks = output["masks"].cpu().numpy() if with_masks and "masks" in output else None

            for i in range(len(scores)):
                if scores[i] < score_threshold:
                    continue
                x1, y1, x2, y2 = boxes[i]
                det = {
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(scores[i]),
                }
                if masks is not None:
                    det["segmentation"] = _encode_mask(masks[i, 0] >= 0.5)
                detections.append(det)

    return detections


def evaluate_coco(model, data_loader, device, ann_json_path: str,
                  image_ids=None, with_masks: bool = False, verbose: bool = True):
    """Standard COCO AP for one split.

    Args:
        ann_json_path: The split's original COCO annotation file, used as
            ground truth.
        image_ids: Restrict scoring to these source COCO image ids — required
            when the dataset dropped images (``skip_empty``) or was capped,
            otherwise the missing images count as pure false negatives.
        with_masks: Also compute ``segm`` AP.

    Returns:
        ``{iou_type: {stat_name: value}}``, e.g.
        ``{"bbox": {"AP": 0.41, "AP50": 0.63, ...}}``. Empty dict if the model
        produced no detections at all.
    """
    if not PYCOCOTOOLS_AVAILABLE:  # pragma: no cover
        raise ImportError("pycocotools is not installed; use engine.evaluate() instead")

    detections = collect_detections(model, data_loader, device, with_masks=with_masks)
    if not detections:
        print("  (no detections produced — skipping COCO evaluation)")
        return {}

    with contextlib.redirect_stdout(_io.StringIO()):
        coco_gt = COCO(ann_json_path)
        coco_dt = coco_gt.loadRes(detections)

    iou_types = ["bbox"] + (["segm"] if with_masks else [])
    results = {}

    for iou_type in iou_types:
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        if image_ids is not None:
            coco_eval.params.imgIds = list(image_ids)

        buffer = _io.StringIO()
        with contextlib.redirect_stdout(buffer):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        results[iou_type] = {
            name: float(coco_eval.stats[i]) for i, name in enumerate(STAT_NAMES)
        }
        if verbose:
            print(f"\n  --- COCO {iou_type} ---")
            print(buffer.getvalue().rstrip())

    return results


def per_category_ap(model, data_loader, device, ann_json_path: str, categories,
                    image_ids=None, iou_type: str = "bbox"):
    """AP@0.5:0.95 per category, for spotting which classes the model fails on.

    Returns:
        ``{category_name: ap}``; a category with no ground truth in this split
        yields ``nan`` and is omitted.
    """
    detections = collect_detections(model, data_loader, device,
                                    with_masks=(iou_type == "segm"))
    if not detections:
        return {}

    with contextlib.redirect_stdout(_io.StringIO()):
        coco_gt = COCO(ann_json_path)
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        if image_ids is not None:
            coco_eval.params.imgIds = list(image_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()

    # precision has shape (iou_thresholds, recall, category, area, max_dets)
    precision = coco_eval.eval["precision"]
    out = {}
    for index, category_id in enumerate(coco_eval.params.catIds):
        values = precision[:, :, index, 0, -1]
        values = values[values > -1]
        if values.size:
            out[categories.get(category_id, str(category_id))] = float(np.mean(values))
    return out
