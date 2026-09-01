"""Tests for the shared COCO dataset wrapper, run against both real datasets.

Every test that needs image/annotation files is parametrized over both
detector configs and skips individually if that dataset isn't present under
``data/raw/`` (the raw data is gitignored).
"""

import json
import os

import numpy as np
import pytest
import torch

from src.detection.car_parts.config import CONFIG as CAR_PARTS
from src.detection.common.coco_dataset import (
    CocoDetectionDataset,
    build_dataset,
    collate_fn,
    polygons_to_mask,
)
from src.detection.damage.config import CONFIG as DAMAGE

CONFIGS = [CAR_PARTS, DAMAGE]
CONFIG_IDS = [c.name for c in CONFIGS]


def _require_dataset(cfg):
    """Skip the calling test unless ``cfg``'s train split is on disk."""
    _, ann_path = cfg.split_paths(cfg.train_split)
    if not os.path.exists(ann_path):
        pytest.skip(f"{cfg.name} dataset not present under data/raw/")
    return ann_path


@pytest.fixture(scope="module")
def datasets():
    """Lazily-built train-split dataset per config, so each is parsed once."""
    cache = {}

    def get(cfg):
        if cfg.name not in cache:
            _require_dataset(cfg)
            cache[cfg.name] = build_dataset(cfg, cfg.train_split)
        return cache[cfg.name]

    return get


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_excluded_categories_are_dropped_and_class_count_matches(cfg, datasets):
    ann_path = _require_dataset(cfg)
    dataset = datasets(cfg)

    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    expected_ids = {
        c["id"] for c in coco["categories"] if c["id"] not in cfg.exclude_category_ids
    }
    assert set(dataset.categories) == expected_ids
    assert dataset.num_classes == max(expected_ids) + 1
    # Class 0 is reserved for background, so no real category may claim it.
    assert 0 not in dataset.categories


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_skips_images_without_annotations(cfg):
    ann_path = _require_dataset(cfg)
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    annotated_image_ids = {
        a["image_id"] for a in coco["annotations"]
        if a["category_id"] not in cfg.exclude_category_ids
        and a["bbox"][2] > 0 and a["bbox"][3] > 0
    }

    dataset = build_dataset(cfg, cfg.train_split, skip_empty=True)
    assert len(dataset) == len(annotated_image_ids)


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_getitem_returns_valid_target(cfg, datasets):
    _require_dataset(cfg)
    dataset = datasets(cfg)
    image, target = dataset[0]

    assert image.dtype == torch.float32
    assert image.ndim == 3 and image.shape[0] == 3

    num_boxes = target["boxes"].shape[0]
    assert num_boxes > 0
    assert target["boxes"].shape == (num_boxes, 4)
    assert target["labels"].shape == (num_boxes,)

    # boxes must be well-formed (x2 > x1, y2 > y1) for torchvision detection models
    x1, y1, x2, y2 = target["boxes"].unbind(1)
    assert torch.all(x2 > x1)
    assert torch.all(y2 > y1)

    assert torch.all(target["labels"] >= 1)
    assert torch.all(target["labels"] < dataset.num_classes)


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_masks_present_only_when_config_asks_for_them(cfg, datasets):
    _require_dataset(cfg)
    dataset = datasets(cfg)
    image, target = dataset[0]

    if not cfg.with_masks:
        assert "masks" not in target
        return

    masks = target["masks"]
    assert masks.dtype == torch.uint8
    assert masks.shape == (target["boxes"].shape[0], image.shape[1], image.shape[2])
    assert set(masks.unique().tolist()) <= {0, 1}
    # A rasterized polygon must cover at least one pixel, otherwise Mask R-CNN
    # trains against an all-zero target for that instance.
    assert torch.all(masks.flatten(1).sum(1) > 0)


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_collate_fn_batches_variable_length_targets(cfg, datasets):
    _require_dataset(cfg)
    dataset = datasets(cfg)
    images, targets = collate_fn([dataset[0], dataset[1]])

    assert len(images) == 2
    assert len(targets) == 2
    assert isinstance(targets[0], dict)


def test_polygons_to_mask_fills_a_square():
    # 10x10 axis-aligned square from (2,2) to (7,7).
    polygon = [[2, 2, 7, 2, 7, 7, 2, 7]]
    mask = polygons_to_mask(polygon, height=10, width=10)

    assert mask.shape == (10, 10)
    assert mask.dtype == np.uint8
    assert mask[4, 4] == 1
    assert mask[0, 0] == 0
    assert mask.sum() > 0


def test_polygons_to_mask_rejects_rle():
    with pytest.raises(TypeError, match="RLE"):
        polygons_to_mask({"counts": [1, 2], "size": [4, 4]}, height=4, width=4)


def test_polygons_to_mask_ignores_degenerate_polygon():
    # Two points is not a polygon; it must contribute no area rather than crash.
    assert polygons_to_mask([[1, 1, 2, 2]], height=5, width=5).sum() == 0


def test_missing_split_name_raises():
    with pytest.raises(KeyError, match="unknown split"):
        CAR_PARTS.split_paths("nonexistent")


def test_dataset_rejects_annotations_with_no_usable_categories(tmp_path):
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps({
        "categories": [{"id": 0, "name": "root"}],
        "images": [], "annotations": [],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="no categories"):
        CocoDetectionDataset(str(tmp_path), str(ann), exclude_category_ids={0})
