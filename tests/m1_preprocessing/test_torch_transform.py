import pytest
import torch

from src.detection_common.trainer import build_arg_parser, build_checkpoint_payload
from src.m1_preprocessing.ap_experiment import plan_arms
from src.m1_preprocessing.torch_transform import (BASELINE, VARIANTS, PreprocessTransform,
                                                  build_preprocess_transform)
from src.m2_parts.config import MASK_CONFIG


# ----------------------------------------------------------------- transform builder
def test_raw_variant_builds_no_transform():
    assert build_preprocess_transform(BASELINE) is None


@pytest.mark.parametrize("variant", [v for v in VARIANTS if v != BASELINE])
def test_named_variant_builds_its_transform(variant):
    transform = build_preprocess_transform(variant)
    assert isinstance(transform, PreprocessTransform)
    assert transform.config == VARIANTS[variant]
    assert repr(transform) == f"M1Preprocess({variant})"


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_preprocess_transform("sharpen")


# --------------------------------------------------------------------- trainer flag
def test_trainer_defaults_to_raw_images():
    assert build_arg_parser(MASK_CONFIG).parse_args([]).preprocess == BASELINE


def test_trainer_accepts_every_variant_and_rejects_others():
    parser = build_arg_parser(MASK_CONFIG)
    for variant in VARIANTS:
        assert parser.parse_args(["--preprocess", variant]).preprocess == variant
    with pytest.raises(SystemExit):
        parser.parse_args(["--preprocess", "sharpen"])


def test_checkpoint_records_the_preprocess_variant():
    class DS:
        num_classes, categories, class_map = 3, {1: "door", 2: "hood"}, None

    args = build_arg_parser(MASK_CONFIG).parse_args(["--preprocess", "specular"])
    payload = build_checkpoint_payload(torch.nn.Linear(1, 1), DS(), MASK_CONFIG, args,
                                       epoch=1, augmented=True)
    assert payload["preprocess"] == "specular"


# ---------------------------------------------------------------- experiment planning
def test_variants_mode_evaluates_one_checkpoint_per_variant_with_raw_first():
    arms, labels = plan_arms("base.pth", None, ["clahe", "raw", "specular"], BASELINE)
    assert arms == [("raw", "base.pth", "raw"), ("clahe", "base.pth", "clahe"),
                    ("specular", "base.pth", "specular")]
    assert labels == {}


def test_retraining_mode_compares_each_model_on_its_own_input():
    arms, labels = plan_arms("base.pth", "m1.pth", ["raw"], BASELINE, trained_variant="specular")
    assert arms == [("raw", "base.pth", "raw"),
                    ("retrained", "m1.pth", "specular"),
                    ("retrained_raw_input", "m1.pth", "raw")]
    assert set(labels) == {"raw", "retrained", "retrained_raw_input"}


def test_baseline_checkpoint_must_be_trained_on_raw_images():
    with pytest.raises(SystemExit):
        plan_arms("m1.pth", None, ["raw"], "specular")


def test_retrained_checkpoint_must_be_trained_with_m1():
    with pytest.raises(SystemExit):
        plan_arms("base.pth", "other.pth", ["raw"], BASELINE, trained_variant=BASELINE)
