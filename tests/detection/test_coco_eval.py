"""Tests for the standard-COCO evaluation path.

The heavy part (running a model over a split) is not exercised here; what is
tested is the glue that is easy to get silently wrong: RLE encoding, the
image-id contract between the dataset and COCOeval, and the fallback when
pycocotools is missing.
"""

import json

import numpy as np
import pytest

from src.detection.car_parts.config import CONFIG as CAR_PARTS
from src.detection.common import coco_eval
from src.detection.common.coco_dataset import build_dataset
from src.detection.damage.config import CONFIG as DAMAGE

CONFIGS = [CAR_PARTS, DAMAGE]
CONFIG_IDS = [c.name for c in CONFIGS]

pytestmark = pytest.mark.skipif(
    not coco_eval.PYCOCOTOOLS_AVAILABLE, reason="pycocotools not installed"
)


def test_stat_names_match_cocoeval_stats_length():
    # COCOeval.stats is a fixed 12-entry array; STAT_NAMES indexes into it.
    assert len(coco_eval.STAT_NAMES) == 12
    assert coco_eval.STAT_NAMES[0] == "AP"
    assert coco_eval.STAT_NAMES[1] == "AP50"


def test_encode_mask_roundtrips_through_rle():
    from pycocotools import mask as mask_utils

    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:15, 10:20] = 1

    rle = coco_eval._encode_mask(mask)
    # counts must be a str, not bytes, or json.dumps on the detections fails.
    assert isinstance(rle["counts"], str)
    assert rle["size"] == [20, 30]

    decoded = mask_utils.decode({**rle, "counts": rle["counts"].encode("utf-8")})
    assert np.array_equal(decoded, mask)
    assert decoded.sum() == 100


def test_encode_mask_accepts_a_boolean_mask():
    # `masks[i, 0] >= 0.5` produces bool, not uint8.
    bool_mask = np.zeros((8, 8), dtype=bool)
    bool_mask[2:4, 2:4] = True
    rle = coco_eval._encode_mask(bool_mask)
    assert rle["size"] == [8, 8]


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_dataset_image_ids_are_source_coco_ids(cfg):
    """COCOeval scores against the original JSON, so ids must match it."""
    _, ann_path = cfg.split_paths(cfg.val_split)
    try:
        dataset = build_dataset(cfg, cfg.val_split)
    except FileNotFoundError:
        pytest.skip(f"{cfg.name} dataset not present under data/raw/")

    with open(ann_path, encoding="utf-8") as f:
        source_ids = {img["id"] for img in json.load(f)["images"]}

    ids = dataset.coco_image_ids
    assert set(ids) <= source_ids
    assert len(ids) == len(dataset)
    # The target must carry that same id, not the index into the dataset.
    _, target = dataset[0]
    assert target["image_id"] == ids[0]


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_coco_image_ids_are_unique(cfg):
    try:
        dataset = build_dataset(cfg, cfg.val_split)
    except FileNotFoundError:
        pytest.skip(f"{cfg.name} dataset not present under data/raw/")

    ids = dataset.coco_image_ids
    assert len(set(ids)) == len(ids)


def test_evaluate_coco_returns_empty_when_model_finds_nothing(monkeypatch):
    monkeypatch.setattr(coco_eval, "collect_detections", lambda *a, **k: [])
    result = coco_eval.evaluate_coco(None, None, None, "unused.json")
    assert result == {}
