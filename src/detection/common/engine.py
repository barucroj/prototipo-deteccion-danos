"""Training and evaluation loops, shared by both detectors.

The evaluation metric here is a from-scratch mAP@0.5 (greedy IoU matching
per class, VOC-style all-points interpolated average precision). It is a
fallback, kept only for environments where ``pycocotools`` cannot be
installed; the standard COCO AP in :mod:`src.detection.common.coco_eval` is
the metric to report. Treat these numbers as directionally useful for
comparing checkpoints, never as comparable to published COCO mAP figures.

The metric is computed on **boxes** for both detectors, including the
mask-producing damage model — mask mAP would need the same matching redone on
mask IoU and is not implemented.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torchvision.ops import box_iou
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, device, scaler=None, max_norm: float = None):
    """Runs one training epoch. Returns the mean total loss across batches.

    Args:
        scaler: Optional ``torch.amp.GradScaler``. When given, the forward
            pass runs under autocast — roughly halves activation memory,
            which is what makes Mask R-CNN on ~1000px CarDD images fit in the
            6 GB of the development GPU.
        max_norm: Optional gradient-norm clipping threshold.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None

    pbar = tqdm(data_loader, desc="Training", leave=True, unit="batch")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device.type, enabled=use_amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            if max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(num_batches, 1)


def _voc_average_precision(recalls, precisions):
    """All-points interpolated average precision (VOC 2010+ style) for one class."""
    recalls = [0.0] + list(recalls) + [1.0]
    precisions = [0.0] + list(precisions) + [0.0]

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        if recalls[i] != recalls[i - 1]:
            ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


@torch.no_grad()
def evaluate(model, data_loader, device, iou_threshold: float = 0.5, score_threshold: float = 0.05):
    """Simplified box mAP@``iou_threshold`` over a validation/test data loader.

    Returns:
        (mean_ap, per_class_ap) where ``per_class_ap`` maps category id to
        its average precision (classes absent from the ground truth are
        omitted).
    """
    model.eval()

    detections_by_class = defaultdict(list)  # label -> [(score, is_true_positive), ...]
    num_ground_truth_by_class = defaultdict(int)

    pbar = tqdm(data_loader, desc="Evaluating", leave=True, unit="batch")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

            for label in gt_labels.tolist():
                num_ground_truth_by_class[label] += 1

            order = torch.argsort(output["scores"], descending=True)
            for i in order.tolist():
                score = output["scores"][i].item()
                if score < score_threshold:
                    continue
                label = output["labels"][i].item()

                candidate_idx = [
                    j for j in range(len(gt_boxes))
                    if gt_labels[j].item() == label and not matched[j]
                ]

                is_true_positive = 0
                if candidate_idx:
                    ious = box_iou(output["boxes"][i : i + 1], gt_boxes[candidate_idx])[0]
                    max_iou, max_pos = ious.max(0)
                    if max_iou.item() >= iou_threshold:
                        matched[candidate_idx[max_pos.item()]] = True
                        is_true_positive = 1

                detections_by_class[label].append((score, is_true_positive))

    per_class_ap = {}
    for label, gt_count in num_ground_truth_by_class.items():
        dets = sorted(detections_by_class.get(label, []), key=lambda d: -d[0])

        tp_cumulative = fp_cumulative = 0
        precisions, recalls = [], []
        for _, is_tp in dets:
            tp_cumulative += is_tp
            fp_cumulative += 1 - is_tp
            precisions.append(tp_cumulative / (tp_cumulative + fp_cumulative))
            recalls.append(tp_cumulative / gt_count)

        per_class_ap[label] = _voc_average_precision(recalls, precisions)

    mean_ap = sum(per_class_ap.values()) / len(per_class_ap) if per_class_ap else 0.0
    return mean_ap, per_class_ap
