"""Forced final class ids (``final_classes``): same ids whatever the dataset labels."""

import json

import pytest

from src.detection_common.coco_dataset import CocoDetectionDataset, apply_class_map, final_categories

MAP = {"left_door": "door", "right_door": "door", "hood": "hood", "roof": None}


def coco(names=("left_door", "right_door", "hood", "roof")):
    cats = [{"id": i, "name": n} for i, n in enumerate(names, start=1)]
    anns = [{"id": i, "image_id": 1, "category_id": i, "bbox": [0, 0, 5, 5], "area": 25,
             "iscrowd": 0, "segmentation": [[0, 0, 5, 0, 5, 5, 0, 5]]} for i in range(1, len(names) + 1)]
    return {"images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            "annotations": anns, "categories": cats}


def test_without_final_classes_ids_come_from_the_map():
    assert final_categories(MAP) == {1: "door", 2: "hood"}


def test_forced_classes_include_the_ones_the_map_never_produces():
    assert final_categories(MAP, ("wheel", "door", "hood", "fender")) == {
        1: "door", 2: "fender", 3: "hood", 4: "wheel"}


def test_a_target_outside_the_forced_classes_raises():
    with pytest.raises(ValueError, match="not in final_classes"):
        final_categories(MAP, ("door",))


def test_apply_class_map_uses_forced_ids_and_counts_absent_classes_as_zero():
    new, stats = apply_class_map(coco(), MAP, final_classes=("door", "fender", "hood"))
    assert [c["name"] for c in new["categories"]] == ["door", "fender", "hood"]
    assert sorted({a["category_id"] for a in new["annotations"]}) == [1, 3]
    assert stats["instances_per_final_class"] == {"door": 2, "fender": 0, "hood": 1}


def test_dataset_num_classes_follows_the_forced_set(tmp_path):
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(coco()), encoding="utf-8")
    ds = CocoDetectionDataset(str(tmp_path), str(path), class_map=MAP,
                              final_classes=("door", "fender", "hood"))
    assert ds.categories == {1: "door", 2: "fender", 3: "hood"} and ds.num_classes == 4
