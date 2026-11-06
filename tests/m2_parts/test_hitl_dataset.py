"""Checks on the HITL Supervisely -> COCO converter (synthetic data, no dataset needed)."""

import json

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from src.detection_common.coco_dataset import polygons_to_mask
from src.m2_parts.hitl_dataset import assign_splits, convert, polygon_area, polygon_mask, polygon_pieces

SQUARE = [[10, 10], [90, 10], [90, 90], [10, 90]]
HOLE = [[40, 40], [60, 40], [60, 60], [40, 60]]


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def test_polygon_area():
    assert polygon_area(SQUARE) == 6400
    assert polygon_area([[0, 0], [1, 1]]) == 0


def test_polygon_without_holes_is_kept_as_is():
    assert polygon_pieces(SQUARE, [], 100, 100) == [[10.0, 10.0, 90.0, 10.0, 90.0, 90.0, 10.0, 90.0]]


@pytest.mark.parametrize("holes", [[HOLE], [HOLE, [[15, 15], [25, 15], [25, 25], [15, 25]]],
                                   [[[30, 20], [70, 20], [70, 30], [30, 30]], [[30, 70], [70, 70], [70, 80], [30, 80]]]])
def test_pieces_reproduce_the_mask_with_holes(holes):
    exact = polygon_mask(SQUARE, holes, 100, 100)
    pieces = polygon_pieces(SQUARE, holes, 100, 100)
    ours = polygons_to_mask(pieces, 100, 100)
    assert iou(ours, exact) > 0.97
    for hole in holes:  # the hole stays empty
        x0, y0 = np.min(hole, 0) + 2
        x1, y1 = np.max(hole, 0) - 2
        assert ours[y0:y1, x0:x1].sum() == 0
    coco = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(pieces, 100, 100)))
    assert iou(coco, exact) > 0.9


def write_dataset(tmp_path, objects_per_image):
    (tmp_path / "File1" / "ann").mkdir(parents=True)
    (tmp_path / "meta.json").write_text(json.dumps(
        {"classes": [{"title": t, "shape": "polygon"} for t in ("Hood", "Front-door")]}), encoding="utf-8")
    for name, objects in objects_per_image.items():
        ann = {"size": {"width": 100, "height": 100}, "objects": objects}
        (tmp_path / "File1" / "ann" / f"{name}.json").write_text(json.dumps(ann), encoding="utf-8")
    return str(tmp_path)


def obj(title, ext, holes=()):
    return {"classTitle": title, "geometryType": "polygon", "points": {"exterior": ext, "interior": list(holes)}}


def test_convert_validates_and_counts(tmp_path):
    root = write_dataset(tmp_path, {
        "a.png": [obj("Hood", SQUARE), obj("Front-door", SQUARE, [HOLE]),
                  obj("Hood", [[1, 1], [2, 2]]), obj("Hood", [[1, 1], [5, 1], [9, 1]])],
        "b.jpg": [obj("Hood", [[1, 1], [2, 2]])],
    })
    coco, stats = convert(root)
    assert [c["name"] for c in coco["categories"]] == ["Front-door", "Hood"]
    assert [im["file_name"] for im in coco["images"]] == ["a.png", "b.jpg"]
    assert len(coco["annotations"]) == 2
    assert stats["dropped"] == {"menos_de_3_puntos": 2, "area_0": 1, "mascara_vacia": 0}
    assert stats["with_holes"] == 1 and stats["holes_iou_min"] > 0.97
    assert stats["images_without_objects"] == 1
    door = next(a for a in coco["annotations"] if a["category_id"] == 1)
    assert door["area"] == 6400 - 400 and door["bbox"] == [10.0, 10.0, 80.0, 80.0]
    assert len(door["segmentation"]) >= 2


def test_ids_are_stable_across_subsets(tmp_path):
    root = write_dataset(tmp_path, {n: [obj("Hood", SQUARE)] for n in ("a.png", "b.png", "c.png")})
    full, _ = convert(root)
    part, _ = convert(root, ["c.png"])
    assert part["images"][0]["id"] == next(im["id"] for im in full["images"] if im["file_name"] == "c.png")


def test_assign_splits_keeps_groups_together_and_respects_forced():
    groups = {f"img{i}": f"g{i // 3}" for i in range(60)}  # 20 groups of 3
    split = assign_splits(groups, seed=0, forced={"g0": "excluido"})
    for g in set(groups.values()):
        assert len({split[i] for i, gg in groups.items() if gg == g}) == 1
    assert {split[i] for i in ("img0", "img1", "img2")} == {"excluido"}
    counts = {s: len({groups[i] for i in split if split[i] == s}) for s in ("train", "val", "test")}
    assert counts == {"train": 15, "val": 2, "test": 2}  # 19 free groups: 80/10/10 rounded
    assert split == assign_splits(groups, seed=0, forced={"g0": "excluido"})


# ---------------------------------------------------------------- door correction (option A)
from src.m2_parts.hitl_dataset import cut_frame, fix_door, mask_pieces  # noqa: E402


def box(x0, y0, x1, y1, size=100):
    m = np.zeros((size, size), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_window_mostly_inside_is_subtracted():
    door, window = box(10, 10, 90, 90), box(20, 15, 80, 40)
    new, n = fix_door(door, [window])
    assert n == 1 and new.sum() == door.sum() - window.sum() and not (new & window).any()


def test_window_mostly_outside_is_kept():
    door, window = box(10, 10, 90, 90), box(5, 0, 95, 15)  # only 5 of 15 rows inside
    new, n = fix_door(door, [window])
    assert n == 0 and (new == door).all()


def test_small_fragments_dropped_and_big_loss_discards():
    door = box(10, 10, 90, 90)
    sliver = box(10, 10, 90, 12)  # 160 px < 5 % of 6400, separated by the window below it
    new, _ = fix_door(door, [box(10, 12, 90, 30)])
    assert not (new & sliver).any() and new.sum() == door.sum() - 18 * 80 - 160
    gone, n = fix_door(door, [box(10, 10, 90, 70)])  # keeps 25 % < 40 %
    assert gone is None and n == 1


def test_mask_pieces_cover_a_mask_with_a_hole():
    mask = box(10, 10, 90, 90) & ~box(40, 40, 60, 60)
    ours = polygons_to_mask(mask_pieces(mask), 100, 100).astype(bool)
    assert iou(ours, mask) > 0.97 and not ours[42:58, 42:58].any()


def test_convert_fixes_doors_and_reports(tmp_path):
    root = write_dataset(tmp_path, {"a.png": [obj("Front-door", SQUARE), obj("Front-window", [[20, 15], [80, 15], [80, 40], [20, 40]]),
                                              obj("Hood", SQUARE)]})
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    meta["classes"].append({"title": "Front-window", "shape": "polygon"})
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    coco, stats = convert(root)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    door = next(a for a in coco["annotations"] if names[a["category_id"]] == "Front-door")
    hood = next(a for a in coco["annotations"] if names[a["category_id"]] == "Hood")
    # window subtracted and the frame above its lower edge (y = 40) cut: rows 41..90 remain
    assert door["area"] == pytest.approx(80 * 50, rel=0.03) and door["bbox"][1] == pytest.approx(41, abs=1)
    fix = stats["door_fix"]
    assert fix["doors"] == 1 and fix["modified"] == 1 and fix["discarded"] == 0
    raw, raw_stats = convert(root, door_fix=False)
    assert "door_fix" not in raw_stats
    assert next(a for a in raw["annotations"] if names[a["category_id"]] == "Front-door")["area"] == hood["area"]


def test_frame_above_the_window_is_cut_across_the_door():
    door, window = box(10, 10, 90, 90), box(30, 15, 70, 40)
    new, n = fix_door(door, [window], is_window=[True], remove_frame=True)
    assert n == 1 and not new[:40].any() and new[40:90, 10:90].all()


def test_slanted_window_side_leaves_no_pillar():
    door = box(0, 0, 100, 100)
    window = np.zeros((100, 100), bool)
    for x in range(20, 80):
        window[10:60 if x < 70 else 60 - 5 * (x - 69), x] = True  # bottom at y=59, right side rising
    new = cut_frame(door, window)
    assert not new[:60].any() and new[60:].all()


def test_mirror_does_not_set_the_edge():
    door, mirror = box(10, 10, 90, 90), box(10, 40, 25, 60)
    new, n = fix_door(door, [mirror], is_window=[False], remove_frame=True)
    assert n == 1 and new[10:40, 30:90].all()


def test_frame_is_kept_without_remove_frame():
    door, window = box(10, 10, 90, 90), box(30, 15, 70, 40)
    new, _ = fix_door(door, [window], is_window=[True])
    assert new[10:15, 10:90].all()
