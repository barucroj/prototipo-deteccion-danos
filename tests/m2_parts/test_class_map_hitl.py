"""Checks on the HITL ten-class map."""

import json
import os

import pytest

from src.detection_common.coco_dataset import apply_class_map, final_categories
from src.m2_parts.class_map_10 import CLASS_MAP_ROBOFLOW, FINAL_CLASSES
from src.m2_parts.class_map_hitl import CLASS_MAP_HITL, CLASS_MAP_MIX, PENDING_DECISIONS, prefixed
from src.m2_parts.hitl_dataset import DEFAULT_ROOT, read_meta

META = os.path.join(DEFAULT_ROOT, "meta.json")
needs_hitl = pytest.mark.skipif(not os.path.isfile(META), reason="HITL dataset not under data/raw/")


@needs_hitl
def test_every_original_class_is_in_the_map():
    names = read_meta(DEFAULT_ROOT)
    assert len(names) == 21
    assert set(CLASS_MAP_HITL) == set(names)


def test_ids_are_the_same_as_roboflow():
    assert final_categories(CLASS_MAP_HITL, FINAL_CLASSES) == final_categories(CLASS_MAP_ROBOFLOW, FINAL_CLASSES)
    assert {v for v in CLASS_MAP_HITL.values() if v is not None} == set(FINAL_CLASSES)


def test_discarded_classes():
    assert {k for k, v in CLASS_MAP_HITL.items() if v is None} == {"License-plate", "Roof", "Rocker-panel"}


def test_pending_decisions_are_final_classes():
    assert set(PENDING_DECISIONS) <= set(FINAL_CLASSES) and "light" not in PENDING_DECISIONS


def test_light_is_headlight_and_tail_light():
    assert {k for k, v in CLASS_MAP_HITL.items() if v == "light"} == {"Headlight", "Tail-light"}


def test_mix_map_is_both_maps_with_prefix():
    assert prefixed({"a": "x", "b": None}, "s") == {"s/a": "x", "s/b": None}
    assert len(CLASS_MAP_MIX) == len(CLASS_MAP_ROBOFLOW) + len(CLASS_MAP_HITL)
    assert CLASS_MAP_MIX["roboflow/back_bumper"] == "bumper" and CLASS_MAP_MIX["hitl/Back-bumper"] == "bumper"
    assert CLASS_MAP_MIX["roboflow/fog_lights"] is None
    assert final_categories(CLASS_MAP_MIX, FINAL_CLASSES) == final_categories(CLASS_MAP_ROBOFLOW, FINAL_CLASSES)


@needs_hitl
def test_map_applies_to_the_converted_dataset():
    from src.m2_parts.hitl_dataset import convert, list_images
    coco, _ = convert(DEFAULT_ROOT, list_images(DEFAULT_ROOT)[:20])
    new, stats = apply_class_map(coco, CLASS_MAP_HITL, final_classes=FINAL_CLASSES)
    assert [c["name"] for c in new["categories"]] == sorted(FINAL_CLASSES)
    assert set(stats["instances_discarded"]) <= {"License-plate", "Roof", "Rocker-panel"}
