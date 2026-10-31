"""Run a rental pair A/B through M1 -> M2 -> M3 -> M4 with each model loaded once.

:class:`PairPipeline` holds the parts and damage Mask R-CNNs; the web
interface creates one and reuses it for every pair.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from src.m1_preprocessing.pipeline import PreprocessConfig, preprocess_file
from src.m1_preprocessing.validation import validate_image
from src.m3_damage.assignment.analyze import (DEFAULT_DAMAGE_THRESHOLD, DEFAULT_MIN_OVERLAP,
                                              DEFAULT_PART_THRESHOLD, ImageAnalysis, analyze_image)
from src.m4_comparison.damages import DEFAULT_DAMAGE_MAX_DISTANCE
from src.m4_comparison.parts import DEFAULT_PART_MAX_DISTANCE
from src.m4_comparison.render import draw_change_map
from src.m4_comparison.report import build_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PARTS_CHECKPOINT = os.path.join(PROJECT_ROOT, "models", "checkpoints", "car_parts", "v2", "best_model.pth")
DEFAULT_DAMAGE_CHECKPOINT = os.path.join(PROJECT_ROOT, "models", "checkpoints", "damage", "v2", "best_model.pth")


@dataclass(frozen=True)
class PipelineSettings:
    """What the user can change between two runs without reloading the models.

    Args:
        preprocess: Optical steps of M1. The current checkpoints were trained
            on raw images, so highlight removal is off by default here.
        validate: Enforce the protocol's minimum resolution (1920x1080).
    """

    preprocess: PreprocessConfig = field(default_factory=lambda: PreprocessConfig(specular=False))
    validate: bool = True
    part_threshold: float = DEFAULT_PART_THRESHOLD
    damage_threshold: float = DEFAULT_DAMAGE_THRESHOLD
    min_overlap: float = DEFAULT_MIN_OVERLAP
    part_max_distance: float = DEFAULT_PART_MAX_DISTANCE
    damage_max_distance: float = DEFAULT_DAMAGE_MAX_DISTANCE


@dataclass
class ImageResult:
    """One photo of the pair after M1 and M3.

    Args:
        image: The BGR image the detectors saw (after M1).
        metadata: M1 metadata (source file, size, transformations).
        analysis: M3 result (report + masks).
        seconds: Wall time of M1 and of M2+M3, ``{"m1": .., "m2_m3": ..}``.
    """

    image: np.ndarray
    metadata: dict
    analysis: ImageAnalysis
    seconds: Dict[str, float]


@dataclass
class PairResult:
    """Both images plus the M4 report and its change map (BGR, drawn on B)."""

    a: ImageResult
    b: ImageResult
    comparison: dict
    change_map: np.ndarray

    def to_json(self) -> dict:
        """JSON-serializable record of the whole pair (no masks or pixels)."""
        return {
            "A": {"m1": self.a.metadata, "m3": self.a.analysis.report, "segundos": self.a.seconds},
            "B": {"m1": self.b.metadata, "m3": self.b.analysis.report, "segundos": self.b.seconds},
            "m4": self.comparison,
        }


class PairPipeline:
    """The orchestrator. Pass loaded models in, or build them with :meth:`from_checkpoints`."""

    def __init__(self, parts_model, part_categories: Dict[int, str],
                 damage_model, damage_categories: Dict[int, str], device,
                 checkpoints: Optional[Dict[str, str]] = None):
        self.parts_model = parts_model
        self.part_categories = part_categories
        self.damage_model = damage_model
        self.damage_categories = damage_categories
        self.device = device
        self.checkpoints = checkpoints

    @classmethod
    def from_checkpoints(cls, parts_checkpoint: str = DEFAULT_PARTS_CHECKPOINT,
                         damage_checkpoint: str = DEFAULT_DAMAGE_CHECKPOINT, device=None) -> "PairPipeline":
        import torch

        from src.detection_common.predict import load_checkpoint

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parts_model, part_categories = load_checkpoint(parts_checkpoint, device)
        damage_model, damage_categories = load_checkpoint(damage_checkpoint, device)
        return cls(parts_model, part_categories, damage_model, damage_categories, device,
                   checkpoints={"partes": parts_checkpoint, "danos": damage_checkpoint})

    def analyze_image(self, path: str, label: str, settings: PipelineSettings,
                      file_name: Optional[str] = None) -> ImageResult:
        """M1 then M2+M3 on one photo file.

        Raises:
            ImageValidationError: when M1 rejects the file.
        """
        start = time.perf_counter()
        pre = preprocess_file(path, settings.preprocess, validate=settings.validate)
        if file_name:
            pre.metadata["archivo"] = file_name
        m1_seconds = time.perf_counter() - start

        start = time.perf_counter()
        analysis = analyze_image(pre.image, self.parts_model, self.part_categories,
                                 self.damage_model, self.damage_categories, self.device,
                                 image_label=label, file_name=pre.metadata["archivo"],
                                 part_threshold=settings.part_threshold,
                                 damage_threshold=settings.damage_threshold,
                                 min_overlap=settings.min_overlap)
        m3_seconds = time.perf_counter() - start
        return ImageResult(image=pre.image, metadata=pre.metadata, analysis=analysis,
                           seconds={"m1": round(m1_seconds, 2), "m2_m3": round(m3_seconds, 2)})

    def analyze_pair(self, path_a: str, path_b: str, settings: Optional[PipelineSettings] = None,
                     name_a: Optional[str] = None, name_b: Optional[str] = None) -> PairResult:
        """Both photos of a rental through every module.

        Both files go through M1 before any detector runs, so a rejected
        photo B does not cost an inference on A.
        """
        settings = settings or PipelineSettings()
        if settings.validate:
            validate_image(path_a)
            validate_image(path_b)

        result_a = self.analyze_image(path_a, "A", settings, file_name=name_a)
        result_b = self.analyze_image(path_b, "B", settings, file_name=name_b)

        report_b = result_b.analysis.report
        comparison = build_report(result_a.analysis.report, report_b, settings.part_max_distance,
                                  settings.damage_max_distance, modelos=self.checkpoints)
        masks_b = {dano["id"]: mask for dano, mask in zip(report_b["danos"], result_b.analysis.damages.masks)}
        change_map = draw_change_map(result_b.image, comparison, masks_b)
        return PairResult(a=result_a, b=result_b, comparison=comparison, change_map=change_map)
