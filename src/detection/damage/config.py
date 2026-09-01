"""Damage detector configuration (thesis module: Deteccion de Danos).

Dataset: ``data/raw/CarDD_release/CarDD_release/CarDD_COCO/`` — CarDD, with 6
damage categories (dent, scratch, crack, glass shatter, lamp broken, tire
flat) at contiguous ids 1-6, so ids are used as-is with 0 reserved for
background -> 7 classes.

Layout differs from the car-parts dataset in both halves: images live in
``{split}2017/`` folders and all three annotation JSONs share a single
``annotations/`` folder. Note the validation split is named ``val``, not
``valid``.

Masks: every CarDD annotation carries a single polygon (verified across all
three splits, no RLE, no crowds), so instance masks are rasterized directly
with cv2 and the detector is a Mask R-CNN. This matters because a scratch is
long, thin and usually diagonal — its bounding box is mostly background, so a
box alone localizes the damage poorly and gives a weak visual for the report.

Class balance is uneven (train instance counts): scratch 2560, dent 1806,
crack 651, lamp broken 494, glass shatter 475, tire flat 225. Expect per-class
AP for ``tire flat`` to be the least reliable number.
"""

from __future__ import annotations

import os

from src.detection.common.config import DetectorConfig

_COCO_ROOT = os.path.join(
    "data", "raw", "CarDD_release", "CarDD_release", "CarDD_COCO"
)

CONFIG = DetectorConfig(
    name="damage",
    description="vehicle damage detector (6 CarDD damage types, with masks)",
    data_root=_COCO_ROOT,
    splits={
        "train": ("train2017", os.path.join("annotations", "instances_train2017.json")),
        "val": ("val2017", os.path.join("annotations", "instances_val2017.json")),
        "test": ("test2017", os.path.join("annotations", "instances_test2017.json")),
    },
    train_split="train",
    val_split="val",
    test_split="test",
    exclude_category_ids=frozenset(),
    with_masks=True,
    arch="mask_rcnn",
    # CarDD is ~8x the car-parts train split (2816 vs 333 images) at ~1000px,
    # so fewer epochs go further and batch size is capped by the 6 GB RTX 3050.
    default_output=os.path.join("models", "checkpoints", "damage", "v1"),
    default_epochs=12,
    default_batch_size=2,
    default_lr=0.005,
)
