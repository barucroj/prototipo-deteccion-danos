"""Checks on the ten-class part maps (car_parts/v3_10c)."""

import json
import os

import pytest
import torch

from src.detection_common.coco_dataset import build_dataset, final_categories
from src.detection_common.trainer import build_arg_parser, build_checkpoint_payload
from src.m2_parts.class_map_10 import (CLASS_MAP_CARPARTS_SEG, CLASS_MAP_ROBOFLOW, FINAL_CLASSES,
                                      MAP_V2_TO_10, PENDING_VISUAL_CHECK)
from src.m2_parts.config import CARPARTS_SEG_10_CONFIG, MASK_10_CONFIG, ROBOFLOW_10_CONFIG

ROBOFLOW_ANNOTATIONS = os.path.join(ROBOFLOW_10_CONFIG.data_root, "train", "_annotations.coco.json")
CARPARTS_SEG = os.path.join("data", "raw", "carparts-seg.ndjson")
needs_roboflow = pytest.mark.skipif(not os.path.isfile(ROBOFLOW_ANNOTATIONS), reason="Roboflow not under data/raw/")


def roboflow_names():
    with open(ROBOFLOW_ANNOTATIONS, encoding="utf-8") as handle:
        return {c["name"] for c in json.load(handle)["categories"] if c["id"] != 0}


@needs_roboflow
def test_every_roboflow_class_is_in_the_map():
    names = roboflow_names()
    assert len(names) == 47
    assert set(CLASS_MAP_ROBOFLOW) == names


def test_final_ids_are_the_ten_classes_contiguous_from_one():
    for class_map in (CLASS_MAP_ROBOFLOW, CLASS_MAP_CARPARTS_SEG):
        cats = final_categories(class_map)
        assert list(cats) == list(range(1, len(cats) + 1))
        assert set(cats.values()) <= set(FINAL_CLASSES)
    assert set(final_categories(CLASS_MAP_ROBOFLOW).values()) == set(FINAL_CLASSES)


def test_pending_entries_are_real_classes():
    assert set(PENDING_VISUAL_CHECK) <= set(CLASS_MAP_ROBOFLOW)


@pytest.mark.skipif(not os.path.isfile(CARPARTS_SEG), reason="carparts-seg.ndjson not under data/raw/")
def test_carparts_seg_map_covers_its_23_classes():
    with open(CARPARTS_SEG, encoding="utf-8") as handle:
        names = set(json.loads(handle.readline())["class_names"].values())
    assert len(names) == 23 and set(CLASS_MAP_CARPARTS_SEG) == names


def test_checkpoint_and_run_info_keep_the_map():
    class DS:
        num_classes = 11
        categories = final_categories(CLASS_MAP_ROBOFLOW)
        class_map = CLASS_MAP_ROBOFLOW
    args = build_arg_parser(MASK_10_CONFIG).parse_args([])
    assert args.output.endswith(os.path.join("car_parts", "v3_10c"))
    payload = build_checkpoint_payload(torch.nn.Linear(1, 1), DS(), MASK_10_CONFIG, args, epoch=1, augmented=True)
    assert payload["class_map"] == CLASS_MAP_ROBOFLOW
    assert payload["num_classes"] == 11


@needs_roboflow
@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_no_discarded_class_reaches_the_model(split):
    ds = build_dataset(ROBOFLOW_10_CONFIG, split)
    assert ds.categories == final_categories(CLASS_MAP_ROBOFLOW) and ds.num_classes == 11
    assert {a["category_id"] for s in ds.samples for a in s[4]} <= set(ds.categories)
    discarded = {name for name, target in CLASS_MAP_ROBOFLOW.items() if target is None}
    assert set(ds.class_map_stats["instances_discarded"]) <= discarded


CARPARTS_SEG_VAL = os.path.join(CARPARTS_SEG_10_CONFIG.data_root, "annotations", "instances_val.json")


def test_both_datasets_get_the_same_ten_ids():
    ten = final_categories(CLASS_MAP_ROBOFLOW, FINAL_CLASSES)
    assert final_categories(CLASS_MAP_CARPARTS_SEG, FINAL_CLASSES) == ten
    assert CARPARTS_SEG_10_CONFIG.final_classes == ROBOFLOW_10_CONFIG.final_classes == FINAL_CLASSES
    assert MASK_10_CONFIG.final_classes == FINAL_CLASSES


@pytest.mark.skipif(not os.path.isfile(CARPARTS_SEG_VAL), reason="carparts_seg not under data/raw/")
def test_carparts_seg_dataset_has_ten_ids_and_no_fender_or_grille():
    ds = build_dataset(CARPARTS_SEG_10_CONFIG, "val")
    assert ds.categories == final_categories(CLASS_MAP_ROBOFLOW, FINAL_CLASSES) and ds.num_classes == 11
    counts = ds.class_map_stats["instances_per_final_class"]
    assert counts["fender"] == 0 and counts["grille"] == 0


def test_light_is_only_headlights_and_taillights():
    """Decided 2026-11-06: fog lights, indicators and back_light are not light."""
    light = {k for k, v in CLASS_MAP_ROBOFLOW.items() if v == "light"}
    assert light == {"left_front_light", "right_front_lights", "left_back_lights", "right_back_light"}
    for name in ("left_fog_light", "right_fog_lights", "fog_lights", "indicator_light", "back_light"):
        assert CLASS_MAP_ROBOFLOW[name] is None


def test_v2_fog_and_indicator_predictions_are_dropped():
    assert MAP_V2_TO_10["fog_light"] is None and MAP_V2_TO_10["indicator_light"] is None
    assert MAP_V2_TO_10["headlight"] == MAP_V2_TO_10["taillight"] == "light"
