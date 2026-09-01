"""Per-dataset configuration for a detector.

Everything that differs between the two detectors in this project lives in a
``DetectorConfig``; the training loop, dataset wrapper, model builder and
evaluation code in this package are dataset-agnostic and driven entirely by
one of these objects. Adding a third detector means adding a config, not
copying a pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple


@dataclass(frozen=True)
class DetectorConfig:
    """Describes one COCO-style detection dataset and how to train on it.

    Args:
        name: Short slug used in log output and as the default checkpoint
            subfolder (e.g. ``"car_parts"``, ``"damage"``).
        description: One-line human description, printed at train start.
        data_root: Dataset root, relative to the project root.
        splits: Maps a logical split name to
            ``(images_subdir, annotations_relpath)``, both relative to
            ``data_root``. The two datasets here disagree on both halves:
            car-parts keeps the JSON inside the split folder, CarDD keeps all
            three JSONs in a shared ``annotations/`` folder.
        train_split / val_split / test_split: Which keys of ``splits`` to use.
        exclude_category_ids: Category ids present in the JSON's ``categories``
            list that are not real foreground classes and must not become
            model classes (Roboflow exports a root/supercategory at id 0).
        with_masks: If True the dataset yields instance masks and ``arch``
            should be a Mask R-CNN variant.
        arch: Key understood by :func:`src.detection.common.model.build_model`.
        default_output: Default ``--output`` checkpoint folder.
        default_epochs / default_batch_size / default_lr: CLI defaults, tuned
            per dataset (CarDD is ~8x larger than the car-parts split).
    """

    name: str
    description: str
    data_root: str
    splits: Dict[str, Tuple[str, str]]
    train_split: str = "train"
    val_split: str = "valid"
    test_split: str = "test"
    exclude_category_ids: FrozenSet[int] = field(default_factory=frozenset)
    with_masks: bool = False
    arch: str = "faster_rcnn"
    default_output: str = ""
    default_epochs: int = 10
    default_batch_size: int = 2
    default_lr: float = 0.005

    def split_paths(self, split: str, data_root: str = None) -> Tuple[str, str]:
        """Absolute-ish ``(images_dir, annotations_path)`` for one split.

        Args:
            split: A key of ``self.splits``.
            data_root: Overrides ``self.data_root`` (from ``--data-root``).
        """
        if split not in self.splits:
            raise KeyError(
                f"{self.name}: unknown split {split!r}; known splits: {sorted(self.splits)}"
            )
        root = data_root if data_root is not None else self.data_root
        images_subdir, ann_relpath = self.splits[split]
        return os.path.join(root, images_subdir), os.path.join(root, ann_relpath)
