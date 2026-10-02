"""Checks on the two detector configs.

These guard the assumptions the shared pipeline makes about each dataset:
the split names it will ask for exist, the class count is what the thesis
expects, and the two configs don't collide on checkpoint output folders.
"""

import os

import pytest

from src.detection.car_parts.config import CONFIG as CAR_PARTS
from src.detection.car_parts.config import MASK_CONFIG as CAR_PARTS_MASKS
from src.detection.common.model import ARCHITECTURES
from src.detection.common.trainer import SELECT_BY, build_arg_parser, validate_args
from src.detection.damage.config import CONFIG as DAMAGE

CONFIGS = [CAR_PARTS, CAR_PARTS_MASKS, DAMAGE]
CONFIG_IDS = ["car_parts", "car_parts_masks", "damage"]


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
    outputs = [c.default_output for c in CONFIGS]
    assert len(set(outputs)) == len(outputs)


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_select_by_is_valid_for_the_config(cfg):
    assert cfg.default_select_by in SELECT_BY
    if cfg.default_select_by == "segm":
        assert cfg.with_masks


def test_car_parts_mask_config_is_the_m3_input():
    # Same dataset as v1, but instance masks, Mask R-CNN and mask-AP selection.
    assert CAR_PARTS_MASKS.name == CAR_PARTS.name
    assert CAR_PARTS_MASKS.data_root == CAR_PARTS.data_root
    assert CAR_PARTS_MASKS.exclude_category_ids == CAR_PARTS.exclude_category_ids
    assert CAR_PARTS_MASKS.with_masks is True
    assert CAR_PARTS_MASKS.arch == "mask_rcnn"
    assert CAR_PARTS_MASKS.default_select_by == "segm"
    # The boxes-only v1 config stays untouched so v1 remains reproducible.
    assert CAR_PARTS.with_masks is False and CAR_PARTS.arch == "faster_rcnn"


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_default_cli_args_are_consistent(cfg):
    validate_args(cfg, build_arg_parser(cfg).parse_args([]))


@pytest.mark.parametrize("flags", [
    ["--select-by", "segm", "--metric", "simple"],
    ["--arch", "faster_rcnn"],
])
def test_contradictory_flags_fail_before_training(flags):
    with pytest.raises(SystemExit):
        validate_args(CAR_PARTS_MASKS, build_arg_parser(CAR_PARTS_MASKS).parse_args(flags))


def test_segm_selection_is_rejected_for_a_boxes_only_config():
    with pytest.raises(SystemExit):
        validate_args(CAR_PARTS, build_arg_parser(CAR_PARTS).parse_args(["--select-by", "segm"]))


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
