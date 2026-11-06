"""Checks on compare_10c: v2 -> ten-class mapping, groups, paired differences, verdicts.

The run of the real v2 against itself is a manual check (needs the checkpoint
and the GPU): ``models/checkpoints/car_parts/v2/compare_10c/``.
"""

import numpy as np
import pytest

from src.detection_common.coco_dataset import final_categories
from src.m2_parts.class_map import CLASS_MAP
from src.m2_parts.class_map_10 import CLASS_MAP_ROBOFLOW, FINAL_CLASSES, MAP_V2_TO_10, compose_maps
from src.m2_parts.compare_10c import (GROUPS, build_report, model_kind, render_markdown,
                                      to_ten_classes, verdict)
from src.m2_parts.compare_versions import per_image_eval

TEN = final_categories(CLASS_MAP_ROBOFLOW, FINAL_CLASSES)
TEN_ID = {n: i for i, n in TEN.items()}


def test_groups_cover_the_ten_classes_once():
    names = [c for classes in GROUPS.values() for c in classes]
    assert sorted(names) == sorted(FINAL_CLASSES)
    assert set(GROUPS["carparts_seg_sin_etiqueta"]) == {"fender", "grille", "glass"}


def test_compose_maps_rejects_a_split_class():
    with pytest.raises(ValueError, match="maps to both"):
        compose_maps({"a": "x", "b": "x"}, {"a": "p", "b": "q"})


def test_v2_map_covers_v2s_29_classes_and_reaches_all_ten():
    assert set(MAP_V2_TO_10) == {v for v in CLASS_MAP.values() if v is not None}
    assert len(MAP_V2_TO_10) == 29
    assert {v for v in MAP_V2_TO_10.values() if v is not None} == set(FINAL_CLASSES)
    assert MAP_V2_TO_10["windshield"] == "glass"  # in v2 this name is the glass (front_glass)


def test_model_kind():
    assert model_kind(TEN, CLASS_MAP_ROBOFLOW) == "ten"
    v2 = {i: n for i, n in enumerate(sorted(MAP_V2_TO_10), start=1)}
    assert model_kind(v2, None) == "v2"
    with pytest.raises(ValueError):
        model_kind({1: "spaceship"}, None)


def box(image_id, cat, x, score):
    return {"image_id": image_id, "category_id": cat, "bbox": [x, 0.0, 10.0, 10.0], "score": score}


def test_v2_front_and_back_door_on_one_object_count_once():
    v2 = {1: "front_door", 2: "back_door", 3: "handle"}
    dets = [box(1, 1, 0, 0.9), box(1, 2, 1, 0.8), box(1, 3, 50, 0.9)]
    out = to_ten_classes(dets, v2, "v2")
    assert len(out) == 1 and out[0]["category_id"] == TEN_ID["door"] and out[0]["score"] == 0.9


def test_ten_class_model_is_only_relabeled_by_name():
    cats = {1: "wheel", 2: "door"}  # other ids, same names
    out = to_ten_classes([box(1, 1, 0, 0.9), box(1, 2, 0, 0.8)], cats, "ten")
    assert [d["category_id"] for d in out] == [TEN_ID["wheel"], TEN_ID["door"]]


def test_verdict_needs_the_interval_to_exclude_zero():
    assert verdict({"low": 0.01, "high": 0.2}) == "mejora"
    assert verdict({"low": -0.2, "high": -0.01}) == "empeora"
    assert verdict({"low": -0.01, "high": 0.2}) == "no distinguible"
    assert verdict({"low": None, "high": None}) == "sin datos"


def synthetic(n_images=6, seed=0):
    rng = np.random.default_rng(seed)
    images, anns, dets = [], [], []
    for i in range(1, n_images + 1):
        images.append({"id": i, "file_name": f"{i}.jpg", "width": 200, "height": 200})
        for cid in (TEN_ID["door"], TEN_ID["glass"], TEN_ID["wheel"]):
            x, y = rng.uniform(0, 120, 2)
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cid, "bbox": [x, y, 40, 40],
                         "area": 1600, "iscrowd": 0,
                         "segmentation": [[x, y, x + 40, y, x + 40, y + 40, x, y + 40]]})
            if rng.random() < 0.8:
                dets.append({"image_id": i, "category_id": cid, "score": float(rng.uniform(0.3, 1)),
                             "bbox": [float(x + rng.normal(0, 4)), float(y), 40.0, 40.0]})
    gt = {"images": images, "annotations": anns,
          "categories": [{"id": i, "name": n} for i, n in TEN.items()]}
    return gt, dets


def test_a_model_against_itself_differs_by_exactly_zero_and_reports_groups():
    gt, dets = synthetic()
    img_ids = [im["id"] for im in gt["images"]]
    pe = per_image_eval(gt, dets, "bbox", list(TEN), img_ids)
    report = build_report({"a/bbox": pe, "b/bbox": pe}, gt, TEN, "a", n_boot=50, seed=0,
                          split="test", dataset="roboflow")
    d = report["differences_vs_reference"]["b"]["bbox"]
    assert d["AP"]["value"] == 0 and d["AP"]["ci95"] == {"low": 0.0, "high": 0.0}
    assert d["AP"]["verdict"] == "no distinguible"
    groups = report["models"]["a"]["bbox"]["groups"]
    per_class = report["models"]["a"]["bbox"]["per_class"]
    assert groups["carparts_seg_sin_etiqueta"]["value"] == pytest.approx(per_class["glass"]["value"])
    assert groups["resto"]["value"] == pytest.approx(
        np.mean([per_class["door"]["value"], per_class["wheel"]["value"]]))
    assert report["instances"]["fender"] == {"count": 0, "unreliable": True}
    assert "## b − a" in render_markdown(report)


def test_external_test_scores_only_the_shared_criterion_classes():
    from src.m2_parts.compare_10c import EXTERNAL_CLASSES, EXTERNAL_GROUPS, EXTERNAL_WARNING
    assert set(EXTERNAL_CLASSES) == {"light", "bumper", "hood", "wheel", "mirror", "trunk"}
    gt, dets = synthetic()
    names = {i: n for i, n in TEN.items() if n in EXTERNAL_CLASSES}
    img_ids = [im["id"] for im in gt["images"]]
    pe = per_image_eval(gt, dets, "bbox", list(names), img_ids)
    report = build_report({"v2/bbox": pe}, gt, names, "v2", n_boot=20, seed=0, split="test",
                          dataset="carparts_seg", groups=EXTERNAL_GROUPS, warning=EXTERNAL_WARNING)
    assert set(report["instances"]) == set(EXTERNAL_CLASSES)
    assert "door" not in report["models"]["v2"]["bbox"]["per_class"]
    assert report["models"]["v2"]["bbox"]["AP"]["value"] == pytest.approx(
        report["models"]["v2"]["bbox"]["per_class"]["wheel"]["value"])  # only wheel has GT here
    assert report["warning"].startswith("PRUEBA EXTERNA")
    assert "PRUEBA EXTERNA" in render_markdown(report)


def test_datasets_share_the_ten_ids_and_hitl_warns():
    from src.m2_parts.compare_10c import DATASETS, HITL_WARNING
    from src.detection_common.coco_dataset import final_categories as fc
    assert set(DATASETS) == {"roboflow", "hitl", "mix", "carparts_seg"}
    ten = {fc(cfg.class_map, cfg.final_classes)[i] for cfg in DATASETS.values() for i in range(1, 11)}
    assert len(ten) == 10
    assert len({tuple(fc(cfg.class_map, cfg.final_classes).items()) for cfg in DATASETS.values()}) == 1
    assert "optimista" in HITL_WARNING
