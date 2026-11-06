"""``--init-from`` / ``--reset-predictors``: start a run from another checkpoint's weights."""

import pytest
import torch

from src.detection_common.model import build_model
from src.detection_common.trainer import (PREDICTOR_PREFIXES, build_arg_parser, load_initial_weights,
                                          validate_args)
from src.m2_parts.config import MASK_10_CONFIG

CATS = {1: "door", 2: "hood"}


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    torch.manual_seed(0)
    model = build_model(3, arch="mask_rcnn", pretrained=False)
    path = tmp_path_factory.mktemp("ckpt") / "best_model.pth"
    torch.save({"model_state_dict": model.state_dict(), "arch": "mask_rcnn", "num_classes": 3,
                "categories": CATS, "epoch": 4, "detector": "car_parts"}, path)
    return str(path), model.state_dict()


def fresh(num_classes=3):
    torch.manual_seed(1)
    return build_model(num_classes, arch="mask_rcnn", pretrained=False)


def test_loads_every_weight_when_the_classes_match(source):
    path, state = source
    model = fresh()
    info = load_initial_weights(model, path, "mask_rcnn", CATS)
    assert info["reinitialized"] == [] and info["source_epoch"] == 4
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())


def test_reset_keeps_the_trunk_and_reinitializes_only_the_predictors(source):
    path, state = source
    model = fresh()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    info = load_initial_weights(model, path, "mask_rcnn", CATS, reset_predictors=True)
    assert info["reinitialized"]
    assert all(k.startswith(PREDICTOR_PREFIXES) for k in info["reinitialized"])
    for k, v in model.state_dict().items():
        expected = before[k] if k.startswith(PREDICTOR_PREFIXES) else state[k]
        assert torch.equal(v, expected), k


def test_other_classes_need_reset_predictors(source):
    path, _ = source
    other = {1: "door", 2: "hood", 3: "wheel"}
    with pytest.raises(SystemExit, match="reset-predictors"):
        load_initial_weights(fresh(4), path, "mask_rcnn", other)
    info = load_initial_weights(fresh(4), path, "mask_rcnn", other, reset_predictors=True)
    assert info["same_classes"] is False


def test_other_arch_is_rejected(source):
    path, _ = source
    model = build_model(3, arch="faster_rcnn", pretrained=False)
    with pytest.raises(SystemExit, match="faster_rcnn"):
        load_initial_weights(model, path, "faster_rcnn", CATS)


def test_reset_predictors_without_init_from_is_rejected():
    args = build_arg_parser(MASK_10_CONFIG).parse_args(["--reset-predictors"])
    with pytest.raises(SystemExit, match="init-from"):
        validate_args(MASK_10_CONFIG, args)


def test_flags_are_off_by_default():
    args = build_arg_parser(MASK_10_CONFIG).parse_args([])
    assert args.init_from is None and args.reset_predictors is False
