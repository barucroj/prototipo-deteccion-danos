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


class SequenceModel:
    """Answers each call with the next model: one output for A, another for B."""

    def __init__(self, models):
        self.models = iter(models)

    def __call__(self, images):
        return next(self.models)(images)


def make_pipeline(damage_rel_boxes=((0.1, 0.1, 0.2, 0.2),)):
    parts = FakeModel([1, 2], [0.95, 0.4], [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)])
    n = len(damage_rel_boxes)
    damage = FakeModel([2] * n, [0.9] * n, list(damage_rel_boxes))
    return PairPipeline(parts, {1: "front_door", 2: "hood"}, damage, {2: "scratch"}, torch.device("cpu"))


def write_image(path, width=320, height=240, seed=0):
    """Blocks of random color: enough texture for the view check to match the same image."""
    blocks = np.random.default_rng(seed).integers(0, 255, (height // 8, width // 8, 3), dtype=np.uint8)
    cv2.imwrite(str(path), cv2.resize(blocks, (width, height), interpolation=cv2.INTER_NEAREST))
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


def test_pair_includes_the_m4_report_and_change_map(tmp_path):
    pair = make_pipeline().analyze_pair(write_image(tmp_path / "a.png"), write_image(tmp_path / "b.png"), SMALL,
                                        name_a="entrega.png", name_b="devolucion.png")
    assert pair.comparison["imagen_a"] == "entrega.png" and pair.comparison["imagen_b"] == "devolucion.png"
    # Same fake detections in both images: the damage is preexisting.
    assert pair.comparison["resumen"] == {"nuevos": 0, "preexistentes": 1, "no_visibles": 0}
    assert pair.change_map.shape == pair.b.image.shape
    assert not np.array_equal(pair.change_map, pair.b.image)


def test_m4_finds_a_new_damage_and_records_the_thresholds(tmp_path):
    pipeline = make_pipeline()
    pipeline.damage_model = SequenceModel([FakeModel([], [], []), FakeModel([2], [0.9], [(0.1, 0.1, 0.2, 0.2)])])
    settings = PipelineSettings(preprocess=PreprocessConfig(specular=False), validate=False,
                                part_max_distance=0.2, damage_max_distance=0.1)
    pair = pipeline.analyze_pair(write_image(tmp_path / "a.png"), write_image(tmp_path / "b.png"), settings)
    assert pair.comparison["resumen"] == {"nuevos": 1, "preexistentes": 0, "no_visibles": 0}
    assert pair.comparison["danos"][0]["id_b"] == "B-01"
    assert pair.comparison["umbral_distancia"] == 0.1 and pair.comparison["umbral_distancia_partes"] == 0.2


def test_unrelated_photos_are_blocked_without_classifying(tmp_path):
    pair = make_pipeline().analyze_pair(write_image(tmp_path / "a.png", seed=1),
                                        write_image(tmp_path / "b.png", seed=2), SMALL)
    assert pair.comparison["verificacion_vista"]["comparable"] is False
    assert pair.comparison["comparacion_forzada"] is False
    assert pair.comparison["danos"] == [] and pair.comparison["mapa_cambios"] is None
    assert pair.change_map is None


def test_blocked_pair_can_be_forced_and_every_damage_needs_review(tmp_path):
    pipeline = make_pipeline()
    pair = pipeline.analyze_pair(write_image(tmp_path / "a.png", seed=1),
                                 write_image(tmp_path / "b.png", seed=2), SMALL)
    comparison, change_map = pipeline.compare(pair.a, pair.b, SMALL, force=True)
    assert comparison["comparacion_forzada"] is True
    assert comparison["resumen"]["preexistentes"] == 1
    assert all(d["requiere_revision"] for d in comparison["danos"])
    assert change_map is not None


def test_forcing_a_comparable_pair_is_not_recorded_as_forced(tmp_path):
    pair = make_pipeline().analyze_pair(write_image(tmp_path / "a.png"), write_image(tmp_path / "b.png"), SMALL,
                                        force=True)
    assert pair.comparison["comparacion_forzada"] is False


def test_report_records_the_checkpoints(tmp_path):
    pipeline = make_pipeline()
    pipeline.checkpoints = {"partes": "parts.pth", "danos": "damage.pth"}
    pair = pipeline.analyze_pair(write_image(tmp_path / "a.png"), write_image(tmp_path / "b.png"), SMALL)
    assert pair.comparison["modelos"] == {"partes": "parts.pth", "danos": "damage.pth"}


def test_to_json_is_serializable(tmp_path):
    pair = make_pipeline().analyze_pair(write_image(tmp_path / "a.png"), write_image(tmp_path / "b.png"), SMALL)
    record = json.loads(json.dumps(pair.to_json()))
    assert set(record) == {"A", "B", "m4"}
    assert record["m4"]["mapa_cambios"] == "mapa_cambios.png"
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
