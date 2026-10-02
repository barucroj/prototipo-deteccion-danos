"""Car-parts detector configuration (thesis module: Segmentacion).

Dataset: ``data/raw/Car parts coco-segmentation/`` — a Roboflow export with
47 part categories. Images sit directly in each split folder next to that
split's ``_annotations.coco.json``.

Category id 0 ("car-parts") is the Roboflow root/supercategory and never
appears on an actual annotation (verified across all three splits), so it is
excluded and the 47 real ids 1-47 double as the model's foreground class ids,
with 0 reserved for background per the torchvision convention -> 48 classes.

Two configs share the dataset:

- ``CONFIG`` — boxes only, Faster R-CNN. What ``car_parts/v1`` was trained
  with; kept so that run stays reproducible.
- ``MASK_CONFIG`` — instance segmentation, Mask R-CNN. This is what module M3
  needs: it assigns each damage to the part maximizing
  ``area(damage_mask & part_mask) / area(damage_mask)``, which a box can only
  approximate. Every annotation in the three splits carries a polygon (8439
  train / 827 valid / 419 test, no RLE), so masks are rasterized exactly.
  ``best_model.pth`` is selected by mask AP, since the masks are the output
  M3 consumes. Train it with ``python -m src.detection.car_parts.train_masks``.
"""

from __future__ import annotations

import os
from dataclasses import replace

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

MASK_CONFIG = replace(
    CONFIG,
    description="car-parts instance segmentation (47 part categories, with masks; M3 input)",
    with_masks=True,
    arch="mask_rcnn",
    default_output=os.path.join("models", "checkpoints", "car_parts", "v2"),
    default_select_by="segm",
)
