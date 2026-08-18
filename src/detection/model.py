"""Faster R-CNN object detector for car parts, built via transfer learning.

Starts from a Faster R-CNN with a MobileNetV3-Large FPN backbone pretrained
on COCO (``torchvision``'s lightest detection model, chosen for CPU-only
training speed over the heavier ResNet50-FPN variant) and replaces its box
head so it predicts the car-parts categories instead of COCO's 91 classes.
With only ~300 training images, fine-tuning this pretrained backbone is far
more viable than training a detector from random init.
"""

from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes: int, pretrained: bool = True):
    """Build a Faster R-CNN detector for ``num_classes`` classes (including background).

    Args:
        num_classes: Total classes the box head should predict, background
            included (e.g. 48 for 47 car-part categories + background).
        pretrained: If True, start from COCO-pretrained weights for both
            the backbone and detection head, then replace the head to
            match ``num_classes``. If False, use a randomly initialized
            backbone (only useful for fast structural tests).

    Returns:
        A ``torchvision`` ``FasterRCNN`` model.
    """
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
