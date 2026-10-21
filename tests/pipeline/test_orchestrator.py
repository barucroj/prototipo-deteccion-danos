"""Orchestrator A/B on real files with fake detectors (no checkpoints, no GPU)."""

import json

import cv2
import numpy as np
import pytest
import torch

from src.m1_preprocessing.pipeline import PreprocessConfig
from src.m1_preprocessing.validation import ImageValidationError
from src.pipeline.orchestrator import PairPipeline, PipelineSettings


class FakeModel:
    """Callable like a torchvision Mask R-CNN in eval mode; boxes in [0, 1] of the image."""

    def __init__(self, labels, scores, rel_boxes):
        self.labels, self.scores, self.rel_boxes = labels, scores, rel_boxes
        self.calls = 0

    def __call__(self, images):
        self.calls += 1
        _, height, width = images[0].shape
        boxes = torch.tensor([[x1 * width, y1 * height, x2 * width, y2 * height]
                              for x1, y1, x2, y2 in self.rel_boxes], dtype=torch.float32).reshape(-1, 4)
        masks = torch.zeros((len(self.labels), 1, height, width))
        for i, (x1, y1, x2, y2) in enumerate(boxes.int().tolist()):
            masks[i, 0, y1:y2, x1:x2] = 1.0
        return [{"labels": torch.tensor(self.labels, dtype=torch.int64),
                 "scores": torch.tensor(self.scores, dtype=torch.float32),
                 "boxes": boxes, "masks": masks}]


def make_pipeline(damage_rel_boxes=((0.1, 0.1, 0.2, 0.2),)):
    parts = FakeModel([1, 2], [0.95, 0.4], [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)])
    n = len(damage_rel_boxes)
    damage = FakeModel([2] * n, [0.9] * n, list(damage_rel_boxes))
    return PairPipeline(parts, {1: "front_door", 2: "hood"}, damage, {2: "scratch"}, torch.device("cpu"))


def write_image(path, width=320, height=240):
    cv2.imwrite(str(path), np.full((height, width, 3), 120, dtype=np.uint8))
    return str(path)


SMALL = PipelineSettings(preprocess=PreprocessConfig(specular=False), validate=False)


def test_pair_runs_m1_and_m3_on_both_images(tmp_path):
    pipeline = make_pipeline()
    pair = pipeline.analyze_pair(write_image(tmp_path / "a.png"), write_image(tmp_path / "b.jpg"), SMALL,
                                 name_a="entrega.png", name_b="devolucion.jpg")

    assert pipeline.parts_model.calls == 2 and pipeline.damage_model.calls == 2
    assert pair.a.analysis.report["imagen"] == "A" and pair.b.analysis.report["imagen"] == "B"
    assert pair.a.metadata["archivo"] == "entrega.png"
    assert pair.b.analysis.report["archivo"] == "devolucion.jpg"
    # The low-confidence part is filtered out by the part threshold (0.7).
    assert [p["clase_parte"] for p in pair.b.analysis.report["partes"]] == ["front_door"]
    dano = pair.b.analysis.report["danos"][0]
    assert dano["id"] == "B-01" and dano["parte"] == "front_door"
    assert pair.comparison is None


def test_to_json_is_serializable(tmp_path):
    pair = make_pipeline().analyze_pair(write_image(tmp_path / "a.png"), write_image(tmp_path / "b.png"), SMALL)
    record = json.loads(json.dumps(pair.to_json()))
    assert set(record) == {"A", "B", "m4"}
    assert record["m4"] is None
    assert set(record["A"]["segundos"]) == {"m1", "m2_m3"}


def test_thresholds_come_from_settings(tmp_path):
    settings = PipelineSettings(preprocess=PreprocessConfig(specular=False), validate=False,
                                part_threshold=0.3, damage_threshold=0.95)
    pair = make_pipeline().analyze_pair(write_image(tmp_path / "a.png"), write_image(tmp_path / "b.png"), settings)
    assert len(pair.a.analysis.report["partes"]) == 2
    assert pair.a.analysis.report["danos"] == []
    assert pair.a.analysis.report["umbral_danos"] == 0.95


def test_low_resolution_is_rejected_before_any_inference(tmp_path):
    pipeline = make_pipeline()
    good = write_image(tmp_path / "a.png", 1920, 1080)
    small = write_image(tmp_path / "b.png")
    with pytest.raises(ImageValidationError) as error:
        pipeline.analyze_pair(good, small, PipelineSettings(preprocess=PreprocessConfig(specular=False)))
    assert error.value.code == "resolucion"
    assert pipeline.parts_model.calls == 0 and pipeline.damage_model.calls == 0
