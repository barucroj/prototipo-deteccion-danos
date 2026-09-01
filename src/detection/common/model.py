"""Detector architectures, built via transfer learning from COCO weights.

Two architectures are available, both on a ResNet50-FPN v2 backbone:

- ``faster_rcnn`` — boxes only. Used for the car-parts detector, where parts
  are large and roughly rectangular so a box is a fine localization.
- ``mask_rcnn`` — boxes + per-instance masks. Used for the damage detector:
  a scratch is long, thin and usually diagonal, so its bounding box is mostly
  background and a box alone is a poor localization of the actual damage.

Both start from COCO-pretrained weights and have their prediction heads
replaced to match this project's class counts. With only hundreds (car parts)
to a few thousand (CarDD) training images, fine-tuning a pretrained backbone
is what makes either of these workable at all.
"""

from __future__ import annotations

from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

ARCHITECTURES = ("faster_rcnn", "mask_rcnn")

#: Hidden channel width of the replacement mask head (torchvision's default).
MASK_PREDICTOR_HIDDEN_DIM = 256


def build_faster_rcnn(num_classes: int, pretrained: bool = True):
    """Faster R-CNN ResNet50-FPN v2 with its box head resized to ``num_classes``."""
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_mask_rcnn(num_classes: int, pretrained: bool = True):
    """Mask R-CNN ResNet50-FPN v2 with both box and mask heads resized."""
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    mask_in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_in_features, MASK_PREDICTOR_HIDDEN_DIM, num_classes
    )
    return model


def build_model(num_classes: int, arch: str = "faster_rcnn", pretrained: bool = True):
    """Build a detector for ``num_classes`` classes (background included).

    Args:
        num_classes: Total classes the heads should predict, background
            included (48 for 47 car parts, 7 for 6 CarDD damage types).
        arch: One of :data:`ARCHITECTURES`.
        pretrained: If True, start from COCO-pretrained weights and then
            replace the heads. If False, use a randomly initialized backbone —
            only useful for fast structural tests and for reloading a
            checkpoint whose weights are about to be overwritten anyway.

    Returns:
        A ``torchvision`` detection model.
    """
    if arch == "faster_rcnn":
        return build_faster_rcnn(num_classes, pretrained)
    if arch == "mask_rcnn":
        return build_mask_rcnn(num_classes, pretrained)
    raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHITECTURES}")
