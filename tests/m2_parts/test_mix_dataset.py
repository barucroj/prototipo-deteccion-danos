"""Checks on the Roboflow + HITL merge (car_parts/v3_10c and v3_mix)."""

import json
import os

import pytest

from src.detection_common.coco_dataset import build_dataset, final_categories
from src.m2_parts.class_map_10 import CLASS_MAP_ROBOFLOW, FINAL_CLASSES
from src.m2_parts.class_map_hitl import CLASS_MAP_MIX
from src.m2_parts.config import MASK_10_CONFIG, MASK_MIX_CONFIG, MIX_10_DATA
from src.m2_parts.mix_dataset import class_counts, merge_coco


def coco(names, images):
    """``images``: {id: [category id, ...]}."""
    anns = [{"id": k, "image_id": i, "category_id": c, "bbox": [0, 0, 5, 5], "area": 25,
             "segmentation": [[0, 0, 5, 0, 5, 5]], "iscrowd": 0}
            for k, (i, c) in enumerate(((i, c) for i, cs in images.items() for c in cs), start=1)]
    return {"images": [{"id": i, "file_name": f"{i}.jpg", "width": 10, "height": 10} for i in images],
            "annotations": anns,
            "categories": [{"id": i, "name": n} for i, n in enumerate(names)]}


ROBO = coco(["car-parts", "hood", "fog_lights"], {1: [1, 2], 2: []})
HITL = coco(["unused", "Hood"], {1: [1], 5: [1, 1]})
ALL = ["hitl/Hood", "roboflow/fog_lights", "roboflow/hood"]


def test_merge_renumbers_and_prefixes():
    out = merge_coco([("roboflow", ROBO, "robo/train", {0}), ("hitl", HITL, "hitl/img", {0})], ALL, seed=None)
    assert [c["name"] for c in out["categories"]] == ALL
    assert [im["id"] for im in out["images"]] == [1, 2, 3, 4]
    assert [im["file_name"] for im in out["images"]] == ["robo/train/1.jpg", "robo/train/2.jpg",
                                                         "hitl/img/1.jpg", "hitl/img/5.jpg"]
    assert [(im["source"], im["source_id"]) for im in out["images"]] == [
        ("roboflow", 1), ("roboflow", 2), ("hitl", 1), ("hitl", 5)]
    names = {c["id"]: c["name"] for c in out["categories"]}
    got = sorted((a["image_id"], names[a["category_id"]]) for a in out["annotations"])
    assert got == [(1, "roboflow/fog_lights"), (1, "roboflow/hood"), (3, "hitl/Hood"), (4, "hitl/Hood"),
                   (4, "hitl/Hood")]
    assert [a["id"] for a in out["annotations"]] == [1, 2, 3, 4, 5]


def test_shuffle_keeps_annotations_with_their_image():
    a = merge_coco([("roboflow", ROBO, "r", {0}), ("hitl", HITL, "h", {0})], ALL, seed=3)
    b = merge_coco([("roboflow", ROBO, "r", {0}), ("hitl", HITL, "h", {0})], ALL, seed=3)
    assert a == b
    by_file = {im["file_name"]: im["id"] for im in a["images"]}
    per_file = {f: sorted(x["category_id"] for x in a["annotations"] if x["image_id"] == i)
                for f, i in by_file.items()}
    assert per_file == {"r/1.jpg": [2, 3], "r/2.jpg": [], "h/1.jpg": [1], "h/5.jpg": [1, 1]}


def test_merge_rejects_unknown_categories_and_excluded_annotations():
    with pytest.raises(ValueError, match="not in all_categories"):
        merge_coco([("roboflow", ROBO, "r", {0})], ["roboflow/hood"])
    bad = coco(["car-parts", "hood"], {1: [0]})
    with pytest.raises(ValueError, match="excluded category"):
        merge_coco([("roboflow", bad, "r", {0})], ["roboflow/hood"])


def test_configs_differ_only_in_the_training_split():
    assert MASK_10_CONFIG.train_split == "train_roboflow" and MASK_MIX_CONFIG.train_split == "train"
    for cfg in (MASK_10_CONFIG, MASK_MIX_CONFIG):
        assert cfg.val_split == "val" and cfg.class_map == CLASS_MAP_MIX and cfg.final_classes == FINAL_CLASSES
        assert cfg.with_masks and cfg.arch == "mask_rcnn" and cfg.default_select_by == "segm"
    assert MASK_10_CONFIG.default_output != MASK_MIX_CONFIG.default_output
    assert final_categories(CLASS_MAP_MIX, FINAL_CLASSES) == final_categories(CLASS_MAP_ROBOFLOW, FINAL_CLASSES)


MIX_DIR = os.path.join(MIX_10_DATA.data_root, "parts_mix_coco")
needs_mix = pytest.mark.skipif(not os.path.isfile(os.path.join(MIX_DIR, "instances_test.json")),
                               reason="parts_mix_coco not built (python -m src.m2_parts.mix_dataset)")


def load(split):
    with open(os.path.join(MIX_DIR, f"instances_{split}.json"), encoding="utf-8") as f:
        return json.load(f)


@needs_mix
def test_mixed_files_have_the_expected_images_and_no_leak():
    files = {s: {im["file_name"] for im in load(s)["images"]} for s in ("train", "train_roboflow", "val", "test")}
    assert files["train_roboflow"] < files["train"]
    assert all(f.startswith("Car parts coco-segmentation/train/") for f in files["train_roboflow"])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not files[a] & files[b]
    counts = {s: class_counts(load(s))["images"] for s in ("train", "val", "test")}
    assert counts == {"train": {"hitl": 802, "roboflow": 333}, "val": {"hitl": 98, "roboflow": 32},
                      "test": {"hitl": 98, "roboflow": 16}}


@needs_mix
@pytest.mark.parametrize("cfg,split", [(MASK_MIX_CONFIG, "train"), (MASK_10_CONFIG, "train_roboflow"),
                                       (MASK_MIX_CONFIG, "val")])
def test_mixed_dataset_loads_with_the_ten_ids(cfg, split):
    ds = build_dataset(cfg, split)
    assert ds.categories == final_categories(CLASS_MAP_ROBOFLOW, FINAL_CLASSES) and ds.num_classes == 11
    image, target = ds[0]
    assert target["masks"].shape[0] == len(target["labels"]) > 0
    for _, file_name, *_ in ds.samples[:: max(1, len(ds.samples) // 20)]:
        assert os.path.isfile(os.path.join(ds.images_dir, file_name))
