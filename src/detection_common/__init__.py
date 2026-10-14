"""Shared detection pipeline for the damage-detection prototype.

Two detectors share this package:

- :mod:`src.m2_parts` — car parts (Roboflow dataset). Localizes *what part of
  the car* a region is.
- :mod:`src.m3_damage.detector` — 6 damage types (CarDD), Mask R-CNN, boxes +
  masks. Localizes *what damage* is present.

Everything dataset-agnostic (dataset wrapper, model builder, train/eval loops,
CLI, drawing) lives here; each detector package is just a
:class:`~src.detection_common.config.DetectorConfig` plus thin CLI wrappers.
"""
