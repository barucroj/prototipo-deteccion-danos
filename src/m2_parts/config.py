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
  M3 consumes. Train it with ``python -m src.m2_parts.train_masks``.
  It applies :data:`src.m2_parts.class_map.CLASS_MAP` (left/right
  merged, rare classes discarded) to every split; ``--no-class-map`` turns it
  off.
"""

from __future__ import annotations

import os
from dataclasses import replace

from src.m2_parts.class_map import CLASS_MAP
from src.m2_parts.class_map_10 import CLASS_MAP_CARPARTS_SEG, CLASS_MAP_ROBOFLOW, FINAL_CLASSES
from src.m2_parts.class_map_hitl import CLASS_MAP_HITL, CLASS_MAP_MIX
from src.detection_common.config import DetectorConfig

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
    class_map=CLASS_MAP,
)

#: Roboflow on the ten classes of :data:`src.m2_parts.class_map_10.CLASS_MAP_ROBOFLOW`.
#: For evaluation (``compare_10c --dataset roboflow``); training goes through
#: the mixed files (:data:`MASK_10_CONFIG`, :data:`MASK_MIX_CONFIG`).
ROBOFLOW_10_CONFIG = replace(
    MASK_CONFIG,
    description="Roboflow car parts on the 10 final classes (evaluation)",
    default_output="",
    class_map=CLASS_MAP_ROBOFLOW,
    final_classes=FINAL_CLASSES,
)

#: Humans in the Loop part images (``hitl_dataset`` output) on the ten classes.
#: For evaluation (``compare_10c --dataset hitl``).
HITL_10_CONFIG = replace(
    ROBOFLOW_10_CONFIG,
    description="Humans in the Loop car parts on the 10 final classes (evaluation)",
    data_root=os.path.join("data", "raw"),
    splits={split: (os.path.join("Car parts and car damages dataset", "Car damages dataset", "File1", "img"),
                    os.path.join("hitl_parts_coco", f"instances_{split}.json"))
            for split in ("train", "val", "test")},
    val_split="val",
    exclude_category_ids=frozenset(),
    class_map=CLASS_MAP_HITL,
)

#: Roboflow + HITL merged by :mod:`src.m2_parts.mix_dataset` (file names
#: relative to ``data/raw``, categories ``roboflow/<name>`` and ``hitl/<name>``).
#: Splits: ``train`` (both), ``train_roboflow`` (Roboflow only), ``val`` and
#: ``test`` (both).
MIX_10_DATA = replace(
    ROBOFLOW_10_CONFIG,
    description="Roboflow + HITL car parts on the 10 final classes",
    data_root=os.path.join("data", "raw"),
    splits={split: ("", os.path.join("parts_mix_coco", f"instances_{split}.json"))
            for split in ("train", "train_roboflow", "val", "test")},
    val_split="val",
    exclude_category_ids=frozenset(),
    class_map=CLASS_MAP_MIX,
)

#: ``car_parts/v3_10c``: trained on Roboflow train only; the epoch is chosen on
#: the val of **both** datasets (Roboflow 32 + HITL 98), like ``v3_mix``, so the
#: two runs differ only in the training images. Train with
#: ``python -m src.m2_parts.train_masks_10c``.
MASK_10_CONFIG = replace(
    MIX_10_DATA,
    description="car-parts instance segmentation, 10 classes, Roboflow train (val: Roboflow + HITL)",
    train_split="train_roboflow",
    default_output=os.path.join("models", "checkpoints", "car_parts", "v3_10c"),
)

#: ``car_parts/v3_mix``: Roboflow train (327) + HITL train (802) in one stage,
#: same val and recipe as ``v3_10c``. Train with ``python -m src.m2_parts.train_masks_mix``.
MASK_MIX_CONFIG = replace(
    MIX_10_DATA,
    description="car-parts instance segmentation, 10 classes, Roboflow + HITL train (val: both)",
    train_split="train",
    default_output=os.path.join("models", "checkpoints", "car_parts", "v3_mix"),
)

#: Ultralytics Carparts-Seg (``ndjson_a_coco.py`` output) on the same ten ids,
#: **only for the external test** of ``compare_10c --dataset carparts_seg``; it is
#: not used for training (decided 2026-11-06): its val/test are augmented copies
#: of its train photos, and door/glass/fog lights follow another labeling
#: criterion. It labels neither fender nor grille, so ``final_classes`` forces
#: the ten ids. Category ids are the ndjson index + 1, so nothing is excluded.
CARPARTS_SEG_10_CONFIG = replace(
    ROBOFLOW_10_CONFIG,
    description="Carparts-Seg on the 10 final classes (external test only, not for training)",
    data_root=os.path.join("data", "raw", "carparts_seg"),
    splits={split: (os.path.join("images", split),
                    os.path.join("annotations", f"instances_{split}.json"))
            for split in ("train", "val", "test")},
    val_split="val",
    exclude_category_ids=frozenset(),
    default_output="",
    class_map=CLASS_MAP_CARPARTS_SEG,
)
