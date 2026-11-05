import json

import numpy as np
import pytest

from src.m3_damage.assignment.analyze import Detections, finish_analysis
from src.m3_damage.assignment.assign import assign_damages
from src.m3_damage.assignment.merge import (NO_MERGE, MergeSettings, mask_box, mask_overlap, merge_detections,
                                            merge_groups)

H, W = 100, 200


def rect(x1, y1, x2, y2):
    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def detections(items):
    """``items`` = [(type, score, (x1, y1, x2, y2))]."""
    masks = np.array([rect(*box) for _, _, box in items]).reshape(-1, H, W)
    return Detections(names=[t for t, _, _ in items], scores=np.array([s for _, s, _ in items], float),
                      boxes=np.array([box for _, _, box in items], float).reshape(-1, 4), masks=masks)


# Two doors side by side on the top 60 rows; rows 60-100 have no part.
PARTS = detections([("front_door", 0.99, (0, 0, 100, 60)), ("front_door", 0.95, (100, 0, 200, 60))])


def analyze(items, merge=MergeSettings()):
    damages = detections(items)
    assignments = assign_damages(damages.masks, PARTS.masks, PARTS.boxes)
    return finish_analysis(PARTS, damages, assignments, "B", W, H, file_name="b.jpg", merge=merge)


def test_mask_overlap_iou_and_containment():
    big, small = rect(0, 0, 40, 10), rect(0, 0, 10, 10)
    iou, containment = mask_overlap(big, small)
    assert iou == pytest.approx(0.25) and containment == 1.0
    assert mask_overlap(rect(0, 0, 10, 10), rect(50, 50, 60, 60)) == (0.0, 0.0)
    assert mask_overlap(np.zeros((H, W), bool), small) == (0.0, 0.0)


def test_mask_box_is_exclusive_like_torchvision():
    assert mask_box(rect(10, 20, 30, 40)) == [10.0, 20.0, 30.0, 40.0]
    assert mask_box(np.zeros((H, W), bool)) is None


def test_duplicate_with_high_iou_is_merged():
    # batch1 p09: the same dent twice, mask IoU 0.91.
    result = analyze([("dent", 0.76, (10, 10, 60, 50)), ("dent", 0.56, (12, 10, 60, 52))])
    danos = result.report["danos"]
    assert len(danos) == 1
    assert danos[0]["id"] == "B-01" and danos[0]["fusionados"] == ["B-01", "B-02"]
    assert danos[0]["confianza"] == 0.76
    assert danos[0]["bbox"] == [10.0, 10.0, 60.0, 52.0]
    assert danos[0]["area_px"] == int((rect(10, 10, 60, 50) | rect(12, 10, 60, 52)).sum())
    assert len(result.raw_damages) == 2 and result.damages.masks.shape == (1, H, W)


def test_contained_detection_is_merged_despite_low_iou():
    # batch1 p01: a short scratch inside a long one, IoU 0.25 but containment ~1.
    result = analyze([("scratch", 0.97, (5, 40, 95, 50)), ("scratch", 0.93, (10, 40, 30, 50))])
    assert [d["fusionados"] for d in result.report["danos"]] == [["B-01", "B-02"]]


def test_same_type_on_different_parts_is_not_merged():
    # batch1 p06: lamps on different part instances stay separate even if they touch.
    result = analyze([("lamp broken", 0.99, (60, 10, 100, 50)), ("lamp broken", 0.70, (100, 10, 140, 50))])
    assert len(result.report["danos"]) == 2
    assert all(d["fusionados"] is None for d in result.report["danos"])


def test_different_types_are_not_merged():
    result = analyze([("dent", 0.9, (10, 10, 60, 50)), ("scratch", 0.8, (10, 10, 60, 50))])
    assert [d["tipo"] for d in result.report["danos"]] == ["dent", "scratch"]


def test_two_unassigned_damages_can_merge():
    # Both below the doors -> both "sin parte", same region.
    result = analyze([("dent", 0.8, (80, 70, 120, 95)), ("dent", 0.6, (82, 70, 120, 97))])
    (dano,) = result.report["danos"]
    assert dano["parte_id"] is None and dano["fusionados"] == ["B-01", "B-02"]


def test_merging_is_transitive_and_keeps_confidence_order():
    items = [("scratch", 0.9, (10, 10, 40, 20)),   # B-01
             ("dent", 0.85, (150, 10, 190, 50)),   # B-02, alone on the right door
             ("scratch", 0.7, (35, 10, 70, 20)),   # B-03, touches B-01 only a little...
             ("scratch", 0.6, (5, 8, 75, 22))]     # B-04 ...but contains both -> one group
    result = analyze(items)
    danos = result.report["danos"]
    assert [d["id"] for d in danos] == ["B-01", "B-02"]  # ids keep their gaps
    assert danos[0]["fusionados"] == ["B-01", "B-03", "B-04"] and danos[1]["fusionados"] is None


def test_merged_damage_position_is_computed_on_the_union_mask():
    # Long scratch x 10-90 holding a short one x 10-30 on the left door (0-100 x 0-60).
    result = analyze([("scratch", 0.9, (10, 20, 90, 30)), ("scratch", 0.8, (10, 20, 30, 30))])
    (dano,) = result.report["danos"]
    assert dano["parte_id"] == "p-01" and dano["solapamiento"] == 1.0
    ys, xs = np.nonzero(rect(10, 20, 90, 30))
    assert dano["pos_relativa"] == [round(xs.mean() / 100, 4), round(ys.mean() / 60, 4)]


def test_disabled_merge_keeps_every_detection():
    items = [("dent", 0.76, (10, 10, 60, 50)), ("dent", 0.56, (12, 10, 60, 52))]
    result = analyze(items, merge=NO_MERGE)
    assert [d["id"] for d in result.report["danos"]] == ["B-01", "B-02"]
    assert result.report["fusion"] == {"activa": False, "iou_minimo": 0.5, "contencion_minima": 0.8}
    assert result.damages is result.raw_damages


def test_thresholds_are_respected():
    items = [("dent", 0.9, (0, 0, 40, 10)), ("dent", 0.8, (20, 0, 60, 10))]  # IoU 1/3, containment 0.5
    assert len(analyze(items).report["danos"]) == 2
    assert len(analyze(items, MergeSettings(min_iou=0.3)).report["danos"]) == 1
    assert len(analyze(items, MergeSettings(min_containment=0.5)).report["danos"]) == 1


def test_report_stays_json_serializable():
    result = analyze([("dent", 0.76, (10, 10, 60, 50)), ("dent", 0.56, (12, 10, 60, 52))])
    record = json.loads(json.dumps(result.report))
    assert record["fusion"]["activa"] is True


def test_no_detections():
    result = analyze([])
    assert result.report["danos"] == [] and result.damages.masks.shape == (0, H, W)


def test_merge_groups_and_detections_directly():
    dets = detections([("dent", 0.5, (0, 0, 10, 10)), ("dent", 0.9, (0, 0, 10, 10))])
    groups = merge_groups(dets.names, dets.boxes, dets.masks, [None, None], MergeSettings())
    assert groups == [[0, 1]]
    names, scores, boxes, masks = merge_detections(dets.names, dets.scores, dets.boxes, dets.masks, groups)
    assert names == ["dent"] and scores.tolist() == [0.9] and masks.shape == (1, H, W)


def test_threshold_filter_happens_before_merging():
    from src.evaluation.sweep import restrict_damages
    # A confident dent and a weak copy that extends it: at 0.5 they merge, at 0.7 the copy is gone.
    result = analyze([("dent", 0.9, (10, 10, 40, 40)), ("dent", 0.6, (10, 10, 40, 50))])
    assert result.report["danos"][0]["bbox"] == [10.0, 10.0, 40.0, 50.0]
    kept = restrict_damages(result, 0.7)
    (dano,) = kept.report["danos"]
    assert dano["bbox"] == [10.0, 10.0, 40.0, 40.0] and dano["fusionados"] is None
    assert kept.report["fusion"]["activa"] is True
