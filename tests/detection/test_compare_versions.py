"""Checks on compare_versions: exactness against pycocotools, mapping, bootstrap.

All synthetic; the run of the real v1 against itself is a manual check (see
the runbook), not a unit test, because it needs the checkpoint and the GPU.
"""

import contextlib
import copy
import io

import numpy as np
import pytest
from pycocotools.cocoeval import COCOeval

from src.detection.car_parts.compare_versions import (
    VAL_WARNING, accumulate, build_report, paired_bootstrap, per_image_eval,
    render_markdown, to_final_detections,
)
from src.detection.common.coco_eval import load_coco_gt


def box_poly(x, y, w, h):
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def synthetic_gt(n_images=6, seed=0):
    rng = np.random.default_rng(seed)
    images, anns = [], []
    for i in range(1, n_images + 1):
        images.append({"id": i, "file_name": f"{i}.jpg", "width": 200, "height": 200})
        for cat in (1, 2, 3):
            for _ in range(rng.integers(0, 3)):
                x, y = rng.uniform(0, 120, 2)
                w, h = rng.uniform(20, 70, 2)
                anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cat,
                             "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
                             "segmentation": box_poly(x, y, w, h)})
    cats = [{"id": c, "name": n} for c, n in ((1, "door"), (2, "hood"), (3, "roof"), (4, "tire"))]
    return {"images": images, "annotations": anns, "categories": cats}


def synthetic_dets(gt, seed=1, jitter=6.0):
    """Noisy copies of most GT boxes plus false positives, with random scores."""
    rng = np.random.default_rng(seed)
    dets = []
    for a in gt["annotations"]:
        if rng.random() < 0.8:
            x, y, w, h = np.array(a["bbox"]) + rng.normal(0, jitter, 4)
            dets.append({"image_id": a["image_id"], "category_id": a["category_id"],
                         "bbox": [float(x), float(y), float(abs(w)), float(abs(h))],
                         "score": float(rng.uniform(0.3, 1.0))})
    for img in gt["images"]:
        for _ in range(2):
            x, y = rng.uniform(0, 150, 2)
            dets.append({"image_id": img["id"], "category_id": int(rng.integers(1, 5)),
                         "bbox": [float(x), float(y), 30.0, 30.0],
                         "score": float(rng.uniform(0.0, 0.9))})
    return dets


def pycocotools_reference(gt, dets, cat_ids, img_ids, iou_type="bbox"):
    coco_gt = load_coco_gt(gt)
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco_gt, coco_gt.loadRes(copy.deepcopy(dets)), iou_type)
        ev.params.catIds, ev.params.imgIds = cat_ids, img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    per_class = {}
    for k, cid in enumerate(ev.params.catIds):
        p = ev.eval["precision"][:, :, k, 0, -1]
        if (p > -1).any():
            per_class[cid] = float(p[p > -1].mean())
    return ev.stats[0], ev.stats[1], per_class


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_accumulate_reproduces_pycocotools_exactly(seed, iou_type):
    gt = synthetic_gt(seed=seed)
    dets = synthetic_dets(gt, seed=seed + 10)
    if iou_type == "segm":
        for d in dets:
            d["segmentation"] = box_poly(*d["bbox"])
    cat_ids, img_ids = [1, 2, 3, 4], [i["id"] for i in gt["images"]]
    ap, ap50, per_class = pycocotools_reference(gt, dets, cat_ids, img_ids, iou_type)
    ours = accumulate(per_image_eval(gt, dets, iou_type, cat_ids, img_ids))
    assert ours["AP"] == pytest.approx(ap, abs=1e-12)
    assert ours["AP50"] == pytest.approx(ap50, abs=1e-12)
    assert {c: v["AP"] for c, v in ours["per_class"].items()} == pytest.approx(per_class, abs=1e-12)
    # Category 4 has detections but no ground truth: excluded, like pycocotools.
    assert 4 not in ours["per_class"]


def test_weights_count_an_image_several_times():
    gt = synthetic_gt()
    dets = synthetic_dets(gt)
    img_ids = [i["id"] for i in gt["images"]]
    pe = per_image_eval(gt, dets, "bbox", [1, 2, 3], img_ids)
    # Duplicating image 1 by hand must equal weight 2 on image 1.
    dup = copy.deepcopy(gt)
    dup["images"].append({"id": 99, "file_name": "99.jpg", "width": 200, "height": 200})
    for a in [a for a in gt["annotations"] if a["image_id"] == 1]:
        dup["annotations"].append({**a, "id": 10_000 + a["id"], "image_id": 99})
    dup_dets = dets + [{**d, "image_id": 99} for d in dets if d["image_id"] == 1]
    ref = accumulate(per_image_eval(dup, dup_dets, "bbox", [1, 2, 3], img_ids + [99]))
    weights = [2] + [1] * (len(img_ids) - 1)
    assert accumulate(pe, weights)["AP"] == pytest.approx(ref["AP"], abs=1e-12)


def test_to_final_detections_merges_with_nms_and_drops_discarded():
    categories = {1: "left_door", 2: "right_door", 3: "rare", 4: "hood"}
    class_map = {"left_door": "door", "right_door": "door", "rare": None, "hood": "hood"}
    dets = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [12, 11, 50, 50], "score": 0.6},   # same object
        {"image_id": 1, "category_id": 2, "bbox": [150, 150, 30, 30], "score": 0.7},  # other door
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 5, 5], "score": 0.99},       # discarded
        {"image_id": 1, "category_id": 4, "bbox": [11, 10, 50, 50], "score": 0.8},   # other class
    ]
    out = to_final_detections(dets, categories, class_map, source_is_final=False, nms_iou=0.5)
    # final ids: door=1, hood=2 (alphabetical)
    assert sorted((d["category_id"], d["score"]) for d in out) == [(1, 0.7), (1, 0.9), (2, 0.8)]


def test_to_final_detections_passes_through_a_model_already_on_final_classes():
    class_map = {"left_door": "door", "right_door": "door", "hood": "hood"}
    dets = [{"image_id": 1, "category_id": 7, "bbox": [0, 0, 5, 5], "score": 0.5}]
    out = to_final_detections(dets, {7: "hood"}, class_map, source_is_final=True)
    assert out[0]["category_id"] == 2


def make_evals(gt, dets_a, dets_b):
    img_ids = [i["id"] for i in gt["images"]]
    return {"v1_original": per_image_eval(gt, dets_a, "bbox", [1, 2, 3], img_ids),
            "v1_mapped": per_image_eval(gt, dets_a, "bbox", [1, 2, 3], img_ids),
            "v2": per_image_eval(gt, dets_b, "bbox", [1, 2, 3], img_ids)}


NAMES = {"original": {1: "door", 2: "hood", 3: "roof"}, "final": {1: "door", 2: "hood", 3: "roof"}}


def test_identical_models_give_an_exactly_zero_difference():
    gt = synthetic_gt()
    dets = synthetic_dets(gt)
    report = build_report(make_evals(gt, dets, dets), {"original": gt, "final": gt}, NAMES,
                          "test", n_boot=200, seed=0)
    diff = report["experiment_B_v2_vs_v1"]["difference_v2_minus_v1"]
    assert diff["AP"] == 0.0 and diff["AP50"] == 0.0
    assert diff["AP_ci95"] == {"low": 0.0, "high": 0.0}
    assert report["warning"] is None


def test_a_better_model_has_a_positive_difference():
    gt = synthetic_gt()
    worse = synthetic_dets(gt, jitter=12.0)
    perfect = [{"image_id": a["image_id"], "category_id": a["category_id"],
                "bbox": a["bbox"], "score": 1.0} for a in gt["annotations"]]
    report = build_report(make_evals(gt, worse, perfect), {"original": gt, "final": gt}, NAMES,
                          "test", n_boot=200, seed=0)
    diff = report["experiment_B_v2_vs_v1"]["difference_v2_minus_v1"]
    assert diff["AP"] > 0 and diff["AP_ci95"]["low"] > 0


def test_bootstrap_is_seeded_and_paired():
    gt = synthetic_gt()
    evals = make_evals(gt, synthetic_dets(gt), synthetic_dets(gt, seed=5))
    a = paired_bootstrap(evals, 50, seed=3)
    b = paired_bootstrap(evals, 50, seed=3)
    assert np.array_equal(a["v2"]["AP"], b["v2"]["AP"])
    evals["v2"].img_ids = evals["v2"].img_ids[:-1]
    with pytest.raises(ValueError, match="paired"):
        paired_bootstrap(evals, 5, seed=0)


def test_valid_split_warns_and_small_classes_are_flagged():
    gt = synthetic_gt()
    dets = synthetic_dets(gt)
    report = build_report(make_evals(gt, dets, dets), {"original": gt, "final": gt}, NAMES,
                          "valid", n_boot=20, seed=0)
    assert report["warning"] == VAL_WARNING
    md = render_markdown(report)
    assert VAL_WARNING in md
    rows = report["experiment_B_v2_vs_v1"]["per_class_final"]
    assert all(r["unreliable"] == (r["instances"] < 10) for r in rows)
    assert "poco fiable" in md
