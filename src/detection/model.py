"""Faster R-CNN object detector for car parts, built via transfer learning.

Starts from a Faster R-CNN with a ResNet50-FPN backbone pretrained on COCO
and replaces its box head so it predicts the car-parts categories instead of
COCO's 91 classes. ResNet50 is heavier than MobileNetV3 but provides better
accuracy, making it ideal for GPU training.
With only ~300 training images, fine-tuning this pretrained backbone is far
more viable than training a detector from random init.
"""

from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
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
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
