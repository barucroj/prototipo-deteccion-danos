"""Checks on the car-parts class map and its application on load.

Synthetic tests always run; the ones on the real Roboflow dataset skip when
it is not under ``data/raw/``.
"""

import json
import os

import pytest
import torch

from src.detection.car_parts.class_map import CLASS_MAP, PENDING_DECISIONS
from src.detection.car_parts.config import CONFIG as CAR_PARTS
from src.detection.car_parts.config import MASK_CONFIG
from src.detection.common.coco_dataset import (
    CocoDetectionDataset, apply_class_map, build_dataset, final_categories,
)
from src.detection.common.coco_eval import load_coco_gt
from src.detection.common.trainer import build_arg_parser, build_checkpoint_payload

DATA_PRESENT = os.path.isdir(MASK_CONFIG.data_root)
needs_data = pytest.mark.skipif(not DATA_PRESENT, reason="car-parts dataset not under data/raw/")


def ann(ann_id, image_id, category_id):
    return {"id": ann_id, "image_id": image_id, "category_id": category_id,
            "bbox": [1, 1, 4, 4], "area": 16, "iscrowd": 0,
            "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]]}


def synthetic_coco():
    return {
        "images": [{"id": i, "file_name": f"{i}.jpg", "width": 8, "height": 8} for i in (1, 2, 3)],
        "categories": [{"id": 0, "name": "root"}, {"id": 1, "name": "left_door"},
                       {"id": 2, "name": "right_door"}, {"id": 3, "name": "hood"},
                       {"id": 4, "name": "rare"}],
        "annotations": [ann(1, 1, 1), ann(2, 1, 2), ann(3, 1, 3), ann(4, 2, 2),
                        ann(5, 3, 4)],  # image 3 only has the discarded class
    }


SYNTH_MAP = {"left_door": "door", "right_door": "door", "hood": "hood", "rare": None}


def test_final_ids_are_contiguous_and_sorted_by_name():
    assert final_categories(SYNTH_MAP) == {1: "door", 2: "hood"}
    cats = final_categories(CLASS_MAP)
    assert list(cats) == list(range(1, len(cats) + 1))
    assert list(cats.values()) == sorted(cats.values())


def test_merge_sums_instances_and_discard_removes_them():
    new, stats = apply_class_map(synthetic_coco(), SYNTH_MAP, exclude_category_ids={0})
    assert [c["name"] for c in new["categories"]] == ["door", "hood"]
    assert stats["instances_per_final_class"] == {"door": 3, "hood": 1}
    assert stats["instances_discarded"] == {"rare": 1}
    assert stats["images_emptied"] == 1
    assert {a["category_id"] for a in new["annotations"]} == {1, 2}


def test_class_map_must_match_the_dataset_names():
    with pytest.raises(ValueError, match="missing"):
        apply_class_map(synthetic_coco(), {"left_door": "door"}, {0})
    with pytest.raises(ValueError, match="unknown"):
        apply_class_map(synthetic_coco(), {**SYNTH_MAP, "lefft_door": "door"}, {0})


def test_dataset_drops_emptied_images_and_never_yields_discarded_classes(tmp_path):
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(synthetic_coco()), encoding="utf-8")
    ds = CocoDetectionDataset(str(tmp_path), str(path), exclude_category_ids={0},
                              class_map=SYNTH_MAP)
    assert ds.categories == {1: "door", 2: "hood"} and ds.num_classes == 3
    assert ds.coco_image_ids == [1, 2]  # image 3 lost its only annotation
    assert ds.class_map_stats["images_emptied"] == 1
    labels = {a["category_id"] for s in ds.samples for a in s[4]}
    assert labels <= set(ds.categories)
    # The remapped dict is valid COCOeval ground truth with the final ids.
    assert sorted(load_coco_gt(ds.coco_gt).getCatIds()) == [1, 2]


def test_class_map_table_is_complete_and_pending_items_are_real_classes():
    assert len(CLASS_MAP) == 47
    assert set(PENDING_DECISIONS) <= set(CLASS_MAP)
    assert MASK_CONFIG.class_map is CLASS_MAP
    assert CAR_PARTS.class_map is None  # v1 stays on the original 47 classes


def test_checkpoint_stores_the_class_map():
    class DS:
        num_classes = 3
        categories = {1: "door", 2: "hood"}
        class_map = SYNTH_MAP
    args = build_arg_parser(MASK_CONFIG).parse_args([])
    payload = build_checkpoint_payload(torch.nn.Linear(1, 1), DS(), MASK_CONFIG, args,
                                       epoch=1, augmented=True)
    assert payload["class_map"] == SYNTH_MAP
    assert payload["categories"] == {1: "door", 2: "hood"}


def test_no_class_map_flag():
    assert build_arg_parser(MASK_CONFIG).parse_args([]).class_map is True
    assert build_arg_parser(MASK_CONFIG).parse_args(["--no-class-map"]).class_map is False


@needs_data
@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_real_splits_load_with_the_class_map(split):
    ds = build_dataset(MASK_CONFIG, split)
    final = final_categories(CLASS_MAP)
    assert ds.categories == final
    assert ds.num_classes == len(final) + 1
    labels = {a["category_id"] for s in ds.samples for a in s[4]}
    assert labels <= set(final)
    # Every instance of a discarded original class is removed. Compared by
    # counts, not names: a final name may reuse a discarded original one
    # (original "windshield" = wipers, discarded; final "windshield" = glass).
    original = build_dataset(MASK_CONFIG, split, use_class_map=False)
    counts = {}
    for s in original.samples:
        for a in s[4]:
            name = original.categories[a["category_id"]]
            counts[name] = counts.get(name, 0) + 1
    discarded = {name for name, target in CLASS_MAP.items() if target is None}
    assert ds.class_map_stats["instances_discarded"] == {n: counts[n] for n in sorted(discarded)
                                                         if counts.get(n)}
    kept = sum(ds.class_map_stats["instances_per_final_class"].values())
    assert kept + sum(ds.class_map_stats["instances_discarded"].values()) == sum(counts.values())


@needs_data
@pytest.mark.parametrize("split, expected", [("train", 174 + 144), ("valid", 16 + 18),
                                             ("test", 5 + 11)])
def test_front_doors_merge_by_summing_instances(split, expected):
    stats = build_dataset(MASK_CONFIG, split).class_map_stats
    assert stats["instances_per_final_class"]["front_door"] == expected


@needs_data
def test_no_class_map_reproduces_the_original_47_classes():
    ds = build_dataset(MASK_CONFIG, "train", use_class_map=False)
    assert len(ds.categories) == 47 and ds.class_map is None
