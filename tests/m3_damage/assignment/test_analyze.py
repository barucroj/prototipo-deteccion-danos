"""M3 report format and drawing, on synthetic detections (no models)."""

import json

import cv2
import numpy as np
import pytest

from src.m3_damage.assignment.analyze import Detections, ImageAnalysis, build_report
from src.m3_damage.assignment.assign import assign_damages
from src.m3_damage.assignment.output import save_analysis
from src.m3_damage.assignment.render import draw_analysis

H, W = 100, 200


def rect(x1, y1, x2, y2):
    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def detections(names, scores, boxes):
    masks = np.array([rect(*box) for box in boxes]) if boxes else np.zeros((0, H, W), dtype=bool)
    return Detections(names=list(names), scores=np.array(scores, dtype=float),
                      boxes=np.array(boxes, dtype=float).reshape(-1, 4), masks=masks)


def analysis_of(parts, damages, label="B"):
    assignments = assign_damages(damages.masks, parts.masks, parts.boxes)
    report = build_report(parts, damages, assignments, label, W, H, file_name="car.jpg")
    return ImageAnalysis(report=report, parts=parts, damages=damages, assignments=assignments)


# Two doors of the same class, a scratch on the right one and a dent on nothing.
PARTS = detections(["front_door", "front_door"], [0.99, 0.95], [[0, 0, 100, 100], [100, 0, 200, 100]])
DAMAGES = detections(["scratch", "dent"], [0.87, 0.6], [[120, 20, 140, 40], [90, 0, 110, 10]])


def test_report_is_json_serializable_and_has_the_spec_fields():
    report = analysis_of(PARTS, DAMAGES).report
    json.dumps(report)
    assert report["imagen"] == "B"
    assert (report["width"], report["height"]) == (W, H)
    assert set(report["danos"][0]) == {"id", "tipo", "confianza", "bbox", "area_px", "parte",
                                       "parte_id", "solapamiento", "pos_relativa"}


def test_damage_ids_carry_the_image_label():
    assert [d["id"] for d in analysis_of(PARTS, DAMAGES, "A").report["danos"]] == ["A-01", "A-02"]


def test_parte_id_tells_same_class_instances_apart():
    scratch = analysis_of(PARTS, DAMAGES).report["danos"][0]
    assert scratch["parte"] == "front_door"
    assert scratch["parte_id"] == "p-02"
    assert scratch["solapamiento"] == 1.0
    assert scratch["pos_relativa"] == pytest.approx([0.295, 0.295])
    assert scratch["area_px"] == 400


def test_damage_split_evenly_goes_to_the_most_confident_part():
    dent = analysis_of(PARTS, DAMAGES).report["danos"][1]
    assert dent["parte_id"] == "p-01"
    assert dent["solapamiento"] == 0.5


def test_unassigned_damage_is_kept_with_null_part():
    no_parts = detections([], [], [])
    report = analysis_of(no_parts, DAMAGES).report
    assert len(report["danos"]) == 2
    assert report["danos"][0]["parte"] is None
    assert report["danos"][0]["parte_id"] is None
    assert report["danos"][0]["pos_relativa"] is None


def test_mismatched_assignments_raise():
    with pytest.raises(ValueError):
        build_report(PARTS, DAMAGES, [], "A", W, H)


def test_draw_analysis_marks_damages_and_leaves_the_input_untouched():
    image = np.full((H, W, 3), 60, dtype=np.uint8)
    drawn = draw_analysis(image, analysis_of(PARTS, DAMAGES))
    assert drawn.shape == image.shape
    assert (image == 60).all()
    assert (drawn != image).any()


def test_save_analysis_writes_json_masks_and_annotated_image(tmp_path):
    image = np.full((H, W, 3), 60, dtype=np.uint8)
    analysis = analysis_of(PARTS, DAMAGES)
    paths = save_analysis(analysis, image, str(tmp_path), "car")

    with open(paths["json"], encoding="utf-8") as handle:
        saved = json.load(handle)
    assert [d["mascara"] for d in saved["danos"]] == ["masks/car_B-01.png", "masks/car_B-02.png"]

    mask = cv2.imread(str(tmp_path / saved["danos"][0]["mascara"]), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (H, W)
    assert int((mask == 255).sum()) == saved["danos"][0]["area_px"]
    assert cv2.imread(paths["annotated"]).shape == image.shape


def test_draw_analysis_handles_an_image_without_damages():
    image = np.full((H, W, 3), 60, dtype=np.uint8)
    drawn = draw_analysis(image, analysis_of(PARTS, detections([], [], [])))
    assert drawn.shape == image.shape
