"""Car-parts detector configuration (thesis module: Segmentacion).

Dataset: ``data/raw/Car parts coco-segmentation/`` — a Roboflow export with
47 part categories. Images sit directly in each split folder next to that
split's ``_annotations.coco.json``.

Category id 0 ("car-parts") is the Roboflow root/supercategory and never
appears on an actual annotation (verified across all three splits), so it is
excluded and the 47 real ids 1-47 double as the model's foreground class ids,
with 0 reserved for background per the torchvision convention -> 48 classes.

Boxes only: car parts are large and roughly rectangular, so a bounding box is
an adequate localization and the lighter Faster R-CNN head is enough.
"""

from __future__ import annotations

import os

from src.detection.common.config import DetectorConfig

CONFIG = DetectorConfig(
    name="car_parts",
    description="car-parts object detector (47 part categories)",
    data_root=os.path.join("data", "raw", "Car parts coco-segmentation"),
    splits={
        "train": ("train", os.path.join("train", "_annotations.coco.json")),
        "valid": ("valid", os.path.join("valid", "_annotations.coco.json")),
        "test": ("test", os.path.join("test", "_annotations.coco.json")),
    },
    train_split="train",
    val_split="valid",
    test_split="test",
    exclude_category_ids=frozenset({0}),
    with_masks=False,
    arch="faster_rcnn",
    default_output=os.path.join("models", "checkpoints", "car_parts", "v1"),
    default_epochs=20,
    default_batch_size=2,
    default_lr=0.005,
)
