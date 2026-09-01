"""Structural checks on both detector architectures.

All of these build the model with ``pretrained=False`` so they need no
download and no dataset, and therefore always run.
"""

import pytest
import torch

from src.detection.common.model import ARCHITECTURES, build_model


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_box_head_matches_num_classes(arch):
    num_classes = 48
    model = build_model(num_classes=num_classes, arch=arch, pretrained=False)

    assert model.roi_heads.box_predictor.cls_score.out_features == num_classes
    assert model.roi_heads.box_predictor.bbox_pred.out_features == num_classes * 4


def test_mask_head_matches_num_classes():
    num_classes = 7  # 6 CarDD damage types + background
    model = build_model(num_classes=num_classes, arch="mask_rcnn", pretrained=False)

    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == num_classes


def test_faster_rcnn_has_no_mask_head():
    model = build_model(num_classes=5, arch="faster_rcnn", pretrained=False)
    assert getattr(model.roi_heads, "mask_predictor", None) is None


def test_unknown_arch_raises():
    with pytest.raises(ValueError, match="unknown arch"):
        build_model(num_classes=5, arch="yolo", pretrained=False)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_forward_eval_mode_runs_on_random_image(arch):
    model = build_model(num_classes=5, arch=arch, pretrained=False)
    model.eval()

    image = torch.rand(3, 100, 100)
    with torch.no_grad():
        output = model([image])[0]

    expected_keys = {"boxes", "labels", "scores"}
    if arch == "mask_rcnn":
        expected_keys.add("masks")
    assert set(output.keys()) == expected_keys
    assert output["boxes"].shape[1] == 4


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_forward_train_mode_computes_losses(arch):
    model = build_model(num_classes=3, arch=arch, pretrained=False)
    model.train()

    image = torch.rand(3, 100, 100)
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    if arch == "mask_rcnn":
        mask = torch.zeros(1, 100, 100, dtype=torch.uint8)
        mask[0, 10:50, 10:50] = 1
        target["masks"] = mask

    loss_dict = model([image], [target])

    assert "loss_classifier" in loss_dict
    if arch == "mask_rcnn":
        assert "loss_mask" in loss_dict
    assert all(torch.is_tensor(v) for v in loss_dict.values())
