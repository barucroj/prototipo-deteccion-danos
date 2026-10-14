import json

import numpy as np
import pytest
import torch

from src.m1_preprocessing.ap_experiment import (BASELINE, VARIANTS, PreprocessTransform,
                                                build_arg_parser, build_report, render_markdown)
from src.m1_preprocessing.pipeline import PreprocessConfig, preprocess_image
from src.m2_parts.compare_versions import per_image_eval

pytest.importorskip("pycocotools")


# ------------------------------------------------------------------------ variants
def test_variants_cover_raw_and_every_step_combination():
    assert VARIANTS[BASELINE] is None
    flags = {name: (cfg.specular, cfg.clahe) for name, cfg in VARIANTS.items() if cfg is not None}
    assert flags == {"specular": (True, False), "clahe": (False, True), "both": (True, True)}


def test_cli_defaults_to_test_split_and_all_variants():
    args = build_arg_parser().parse_args(["--detector", "damage"])
    assert args.split == "test" and args.variants == list(VARIANTS)
    assert args.bootstrap == 1000 and args.seed == 0


# ----------------------------------------------------------------------- transform
def _tensor(image_rgb_uint8):
    return torch.from_numpy(image_rgb_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)


def test_transform_with_no_steps_is_an_exact_round_trip():
    rng = np.random.default_rng(0)
    image = _tensor(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8))
    target = {"boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}
    out, out_target = PreprocessTransform(PreprocessConfig(specular=False, clahe=False))(image, target)
    assert torch.equal(out, image)
    assert out_target is target


def test_transform_matches_the_pipeline_and_handles_the_channel_order():
    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    rgb[:] = (170, 30, 30)  # red body, RGB
    rgb[10:25, 20:40] = (255, 250, 250)  # glare
    config = PreprocessConfig(specular=True, clahe=True)

    out, _ = PreprocessTransform(config)(_tensor(rgb), {})

    expected_bgr, _ = preprocess_image(np.ascontiguousarray(rgb[:, :, ::-1]), config)
    expected = _tensor(np.ascontiguousarray(expected_bgr[:, :, ::-1]))
    assert out.shape == (3, 40, 60) and out.dtype == torch.float32
    assert torch.equal(out, expected)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_transform_records_one_timing_per_image():
    transform = PreprocessTransform(PreprocessConfig())
    for _ in range(3):
        transform(_tensor(np.zeros((8, 8, 3), np.uint8)), {})
    assert len(transform.seconds) == 3 and all(s >= 0 for s in transform.seconds)


# -------------------------------------------------------------------------- report
N_IMAGES = 12


def _ground_truth():
    """One 20x20 box of class 1 and one of class 2 in each of 12 images."""
    images = [{"id": i, "width": 100, "height": 100, "file_name": f"{i}.jpg"} for i in range(N_IMAGES)]
    annotations = []
    for i in range(N_IMAGES):
        for cat, x in ((1, 10), (2, 60)):
            annotations.append({"id": len(annotations) + 1, "image_id": i, "category_id": cat,
                                "bbox": [x, 10, 20, 20], "area": 400, "iscrowd": 0})
    return {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "dent"}, {"id": 2, "name": "scratch"}]}


def _detections(miss_class_2_in=()):
    out = []
    for i in range(N_IMAGES):
        out.append({"image_id": i, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9})
        if i not in miss_class_2_in:
            out.append({"image_id": i, "category_id": 2, "bbox": [60, 10, 20, 20], "score": 0.9})
    return out


def _report(variant_detections):
    gt = _ground_truth()
    img_ids, cat_ids = list(range(N_IMAGES)), [1, 2]
    evals = {"bbox": {name: per_image_eval(gt, dets, "bbox", cat_ids, img_ids)
                      for name, dets in variant_detections.items()}}
    meta = {"detector": "damage", "checkpoint": "x.pth", "epoch": 1, "split": "test",
            "images": N_IMAGES, "bootstrap": 200, "seed": 0,
            "preprocess_ms": {"worse": {"median": 12.0, "mean": 13.0, "max": 20.0}}}
    return build_report(evals, gt, {1: "dent", 2: "scratch"}, 200, 0, meta)


def test_identical_variant_has_zero_difference_and_is_not_distinguishable():
    block = _report({BASELINE: _detections(), "same": _detections()})["metrics"]["bbox"]
    assert block["variants"][BASELINE]["AP"] == pytest.approx(1.0)
    diff = block["differences"]["same"]
    assert diff["AP"] == 0.0
    assert diff["AP_ci95"] == {"low": 0.0, "high": 0.0}
    assert diff["AP_distinguible"] is False and diff["AP50_distinguible"] is False


def test_variant_that_loses_detections_is_reported_as_worse():
    block = _report({BASELINE: _detections(),
                     "worse": _detections(miss_class_2_in=range(6))})["metrics"]["bbox"]
    diff = block["differences"]["worse"]
    assert diff["AP"] < 0 and diff["AP_ci95"]["high"] < 0
    assert diff["AP_distinguible"] is True

    rows = {row["clase"]: row for row in block["per_class"]}
    assert rows["dent"][BASELINE] == pytest.approx(1.0) and rows["dent"]["worse"] == pytest.approx(1.0)
    assert rows["scratch"]["worse"] < rows["scratch"][BASELINE]
    assert rows["scratch"]["instancias"] == N_IMAGES and rows["scratch"]["poco_fiable"] is False


def test_baseline_is_required():
    gt = _ground_truth()
    evals = {"bbox": {"clahe": per_image_eval(gt, _detections(), "bbox", [1, 2], list(range(N_IMAGES)))}}
    with pytest.raises(ValueError):
        build_report(evals, gt, {1: "dent", 2: "scratch"}, 10, 0, {})


def test_report_is_json_serializable_and_renders_to_markdown():
    report = _report({BASELINE: _detections(), "worse": _detections(miss_class_2_in=range(6))})
    json.dumps(report)
    text = render_markdown(report)
    assert "## AP de caja" in text and "sin preprocesamiento" in text
    assert "distinguible" in text and "| scratch | 12 |" in text
    assert "Tiempo del preprocesamiento" in text
