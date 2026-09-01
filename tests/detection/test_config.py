"""Checks on the two detector configs.

These guard the assumptions the shared pipeline makes about each dataset:
the split names it will ask for exist, the class count is what the thesis
expects, and the two configs don't collide on checkpoint output folders.
"""

import os

import pytest

from src.detection.car_parts.config import CONFIG as CAR_PARTS
from src.detection.common.model import ARCHITECTURES
from src.detection.damage.config import CONFIG as DAMAGE

CONFIGS = [CAR_PARTS, DAMAGE]
CONFIG_IDS = [c.name for c in CONFIGS]


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_declared_splits_exist_in_the_split_map(cfg):
    for split in (cfg.train_split, cfg.val_split, cfg.test_split):
        assert split in cfg.splits, f"{cfg.name}: {split!r} missing from splits"


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_arch_is_known_and_consistent_with_masks(cfg):
    assert cfg.arch in ARCHITECTURES
    # Only Mask R-CNN can consume the masks the dataset would produce.
    if cfg.with_masks:
        assert cfg.arch == "mask_rcnn"


def test_configs_write_to_different_checkpoint_folders():
    assert CAR_PARTS.default_output != DAMAGE.default_output


def test_car_parts_excludes_the_roboflow_root_category():
    assert CAR_PARTS.exclude_category_ids == frozenset({0})


def test_damage_uses_masks_because_scratches_are_thin():
    assert DAMAGE.with_masks is True
    assert DAMAGE.exclude_category_ids == frozenset()
    # CarDD names its validation split "val", not "valid" like the parts dataset.
    assert DAMAGE.val_split == "val"


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_split_paths_are_built_under_the_data_root(cfg):
    images_dir, ann_path = cfg.split_paths(cfg.train_split)
    assert images_dir.startswith(cfg.data_root)
    assert ann_path.startswith(cfg.data_root)
    assert ann_path.endswith(".json")


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_split_paths_honour_a_data_root_override(cfg):
    images_dir, ann_path = cfg.split_paths(cfg.train_split, data_root="elsewhere")
    assert images_dir.startswith("elsewhere" + os.sep)
    assert ann_path.startswith("elsewhere" + os.sep)
