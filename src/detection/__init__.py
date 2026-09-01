"""Object detection for the damage-detection prototype.

Two detectors share one pipeline:

- :mod:`src.detection.car_parts` — 47 car part categories (Roboflow dataset),
  Faster R-CNN, boxes only. Localizes *what part of the car* a region is.
- :mod:`src.detection.damage` — 6 damage types (CarDD), Mask R-CNN, boxes +
  masks. Localizes *what damage* is present.

Everything dataset-agnostic (dataset wrapper, model builder, train/eval loops,
CLI, drawing) lives in :mod:`src.detection.common`; each detector package is
just a :class:`~src.detection.common.config.DetectorConfig` plus two thin CLI
wrappers.
"""
