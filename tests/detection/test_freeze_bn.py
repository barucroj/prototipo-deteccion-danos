"""Checks on --freeze-bn: BatchNorm running stats stay fixed during training.

Runs one real ``train_one_epoch`` step of a randomly initialized Mask R-CNN on
CPU (tiny synthetic images, internal resize shrunk to keep it fast) and
compares BatchNorm running stats before and after.
"""

import pytest
import torch
from torch import nn

from src.detection.common.engine import train_one_epoch
from src.detection.common.model import build_model, freeze_batchnorm_stats
from src.detection.common.trainer import build_arg_parser
from src.detection.car_parts.config import MASK_CONFIG

# One BN per region of the network: frozen stem, frozen layer1 (weights), later
# backbone stage, FPN and both ROI heads.
WATCHED = (
    "backbone.body.bn1",
    "backbone.body.layer1.0.bn1",
    "backbone.body.layer3.0.bn1",
    "backbone.fpn.inner_blocks.0.1",
    "roi_heads.box_head.0.1",
    "roi_heads.mask_head.0.1",
)


def small_model():
    torch.manual_seed(0)
    model = build_model(3, arch="mask_rcnn", pretrained=False)
    model.transform.min_size = (128,)
    model.transform.max_size = 160
    return model


def one_batch_loader():
    images, targets = [], []
    for _ in range(2):
        images.append(torch.rand(3, 128, 160))
        mask = torch.zeros(1, 128, 160, dtype=torch.uint8)
        mask[0, 20:100, 30:130] = 1
        targets.append({"boxes": torch.tensor([[30.0, 20.0, 130.0, 100.0]]),
                        "labels": torch.tensor([1]), "masks": mask})
    return [(images, targets)]


def bn_stats(model):
    modules = dict(model.named_modules())
    return {name: (modules[name].running_mean.clone(), modules[name].running_var.clone())
            for name in WATCHED}


def stats_changed(before, after):
    return {name: not (torch.equal(before[name][0], after[name][0])
                       and torch.equal(before[name][1], after[name][1]))
            for name in WATCHED}


def run_one_step(freeze_bn):
    model = small_model()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    before = bn_stats(model)
    train_one_epoch(model, optimizer, one_batch_loader(), torch.device("cpu"),
                    freeze_bn=freeze_bn)
    return model, stats_changed(before, bn_stats(model))


def test_running_stats_do_not_change_with_freeze_bn():
    model, changed = run_one_step(freeze_bn=True)
    assert not any(changed.values()), changed
    assert all(not m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    # Only BN is switched: the rest of the model is still in training mode.
    assert model.training and model.rpn.training and model.roi_heads.training


def test_running_stats_change_without_freeze_bn():
    _, changed = run_one_step(freeze_bn=False)
    assert all(changed.values()), changed


def test_freeze_batchnorm_stats_counts_all_69_layers():
    model = small_model()
    model.train()
    assert freeze_batchnorm_stats(model) == 69


@pytest.mark.parametrize("flags, expected", [([], False), (["--freeze-bn"], True)])
def test_freeze_bn_flag_is_off_by_default(flags, expected):
    assert build_arg_parser(MASK_CONFIG).parse_args(flags).freeze_bn is expected
