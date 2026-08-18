"""Training and evaluation loops for the car-parts object detector.

The evaluation metric is a from-scratch mAP@0.5 (greedy IoU matching per
class, VOC-style all-points interpolated average precision). It is a
simplified stand-in for the standard COCO mAP (which averages over IoU
thresholds 0.5:0.95 via ``pycocotools``): ``pycocotools`` needs a C
extension build and isn't installed in this project's environment (see
CLAUDE.md environment notes), so this avoids adding a fragile dependency.
Treat the numbers as directionally useful for comparing checkpoints, not as
a benchmark comparable to published COCO mAP figures.
"""

from collections import defaultdict

import torch
from torchvision.ops import box_iou


def train_one_epoch(model, optimizer, data_loader, device):
    """Runs one training epoch. Returns the mean total loss across batches."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

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
    """Simplified mAP@``iou_threshold`` over a validation/test data loader.

    Returns:
        (mean_ap, per_class_ap) where ``per_class_ap`` maps category id to
        its average precision (classes absent from the ground truth are
        omitted).
    """
    model.eval()

    detections_by_class = defaultdict(list)  # label -> [(score, is_true_positive), ...]
    num_ground_truth_by_class = defaultdict(int)

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
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
