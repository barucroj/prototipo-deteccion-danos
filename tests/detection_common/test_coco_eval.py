"""Tests for the standard-COCO evaluation path.

The heavy part (running a model over a split) is not exercised here; what is
tested is the glue that is easy to get silently wrong: RLE encoding, the
image-id contract between the dataset and COCOeval, and the fallback when
pycocotools is missing.
"""

import json

import numpy as np
import pytest

from src.m2_parts.config import CONFIG as CAR_PARTS
from src.detection_common import coco_eval
from src.detection_common.coco_dataset import build_dataset
from src.m3_damage.detector.config import CONFIG as DAMAGE

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


def test_segm_evaluation_does_not_mutate_the_dataset_ground_truth(tmp_path):
    """Regression: COCOeval(segm) rewrites GT polygons to RLE in place.

    The trainer evaluates against ``dataset.coco_gt``, whose annotation dicts
    the dataset also yields from; without the copy in ``load_coco_gt`` epoch 2
    crashed in ``polygons_to_mask`` with "RLE segmentation is not supported".
    """
    import contextlib
    import io

    from PIL import Image
    from pycocotools.cocoeval import COCOeval

    from src.detection_common.coco_dataset import CocoDetectionDataset

    Image.new("RGB", (20, 20)).save(tmp_path / "a.jpg")
    poly = [[2, 2, 12, 2, 12, 12, 2, 12]]
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.jpg", "width": 20, "height": 20}],
        "categories": [{"id": 1, "name": "door"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 10, 10],
                         "area": 100, "iscrowd": 0, "segmentation": poly}],
    }), encoding="utf-8")
    ds = CocoDetectionDataset(str(tmp_path), str(ann_path), with_masks=True,
                              class_map={"door": "door"})

    for _ in range(2):  # two "epochs" of validation
        gt = coco_eval.load_coco_gt(ds.coco_gt)
        dets = [{"image_id": 1, "category_id": 1, "bbox": [2, 2, 10, 10], "score": 1.0,
                 "segmentation": poly}]
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, gt.loadRes(dets), "segm")
            ev.evaluate()
        assert isinstance(ds.samples[0][4][0]["segmentation"], list)
        _, target = ds[0]
        assert target["masks"].shape == (1, 20, 20)


def _two_class_gt():
    return {
        "images": [{"id": 1, "width": 100, "height": 100}, {"id": 2, "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "door"}, {"id": 2, "name": "hood"}, {"id": 3, "name": "trunk"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [50, 50, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [60, 10, 30, 20], "area": 600, "iscrowd": 0},
        ],
    }


def test_category_stats_per_class_with_missing_class():
    import contextlib
    import io

    from pycocotools.cocoeval import COCOeval

    gt = coco_eval.load_coco_gt(_two_class_gt())
    dets = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.9},  # door found
            {"image_id": 1, "category_id": 2, "bbox": [0, 70, 20, 20], "score": 0.8}]   # hood missed
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.evaluate()
        ev.accumulate()
    stats = coco_eval.category_stats(ev)
    assert set(stats) == {"door", "hood", "trunk"}
    assert stats["door"]["instances"] == 2 and stats["door"]["AR_100"] == pytest.approx(0.5)
    assert 0 < stats["door"]["AP"] <= 0.51 and stats["door"]["AP50"] == pytest.approx(stats["door"]["AP"])
    assert stats["hood"] == {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR_100": 0.0, "instances": 1}
    assert stats["trunk"] == {"AP": None, "AP50": None, "AP75": None, "AR_100": None, "instances": 0}


def test_evaluate_coco_fills_per_class_from_the_same_evaluation(monkeypatch):
    dets = [{"image_id": i, "category_id": 1, "bbox": b, "score": 0.9}
            for i, b in ((1, [10, 10, 30, 30]), (2, [50, 50, 30, 30]))]
    monkeypatch.setattr(coco_eval, "collect_detections", lambda *a, **k: dets)
    per_class = {}
    result = coco_eval.evaluate_coco(None, None, None, _two_class_gt(), verbose=False, per_class=per_class)
    assert set(per_class) == {"bbox"}
    assert per_class["bbox"]["door"]["AP"] == pytest.approx(1.0)
    assert per_class["bbox"]["hood"]["AP"] == 0.0
    # The summary AP is the mean over classes with ground truth.
    assert result["bbox"]["AP"] == pytest.approx(0.5)


def test_format_per_class_marks_classes_without_ground_truth():
    from src.detection_common.trainer import format_per_class

    text = format_per_class({"hood": {"AP": 0.25}, "door": {"AP": 0.6123}, "trunk": {"AP": None}})
    assert text == "door 0.612  hood 0.250  trunk -"
