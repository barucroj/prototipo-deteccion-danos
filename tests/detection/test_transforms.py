"""Tests for training-time augmentation.

The point of these is the silent failure mode: an augmentation that moves the
image but not its labels trains the model against wrong targets and reports no
error at all. So the flip tests assert the boxes and masks actually follow the
pixels, not merely that shapes are preserved.
"""

import random

import pytest
import torch

from src.detection.car_parts.config import CONFIG as CAR_PARTS
from src.detection.common.coco_dataset import build_dataset
from src.detection.common.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomPhotometricJitter,
    build_train_transforms,
)
from src.detection.damage.config import CONFIG as DAMAGE


def _sample(width=100, height=60):
    """An image with one bright block, plus a box and mask around it."""
    image = torch.zeros(3, height, width)
    image[:, 20:40, 10:30] = 1.0  # rows 20-40, cols 10-30

    mask = torch.zeros(1, height, width, dtype=torch.uint8)
    mask[0, 20:40, 10:30] = 1

    target = {
        "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        "labels": torch.tensor([1]),
        "masks": mask,
    }
    return image, target


def test_flip_moves_boxes_to_the_mirrored_position():
    image, target = _sample(width=100)
    flipped_image, flipped_target = RandomHorizontalFlip(p=1.0)(image, target)

    # x spans 10..30 in a 100-wide image, so mirrored it spans 70..90.
    assert flipped_target["boxes"].tolist() == [[70.0, 20.0, 90.0, 40.0]]
    # y must not move.
    assert flipped_image.shape == image.shape


def test_flip_keeps_the_box_on_top_of_the_object():
    """The real invariant: after flipping, the box still bounds the bright pixels."""
    image, target = _sample(width=100)
    flipped_image, flipped_target = RandomHorizontalFlip(p=1.0)(image, target)

    x1, y1, x2, y2 = [int(v) for v in flipped_target["boxes"][0]]
    inside = flipped_image[:, y1:y2, x1:x2]
    assert inside.min() == 1.0, "box no longer covers the object"

    outside = flipped_image.clone()
    outside[:, y1:y2, x1:x2] = 0.0
    assert outside.max() == 0.0, "object pixels found outside the box"


def test_flip_keeps_the_mask_aligned_with_the_box():
    image, target = _sample(width=100)
    _, flipped = RandomHorizontalFlip(p=1.0)(image, target)

    mask = flipped["masks"][0]
    ys, xs = torch.nonzero(mask, as_tuple=True)
    x1, y1, x2, y2 = flipped["boxes"][0]

    assert xs.min().item() == int(x1)
    assert xs.max().item() == int(x2) - 1
    assert ys.min().item() == int(y1)
    assert ys.max().item() == int(y2) - 1


def test_flipping_twice_restores_the_original():
    image, target = _sample()
    flip = RandomHorizontalFlip(p=1.0)

    once_image, once_target = flip(image, target)
    twice_image, twice_target = flip(once_image, once_target)

    assert torch.equal(twice_image, image)
    assert torch.equal(twice_target["boxes"], target["boxes"])
    assert torch.equal(twice_target["masks"], target["masks"])


def test_flip_with_zero_probability_is_a_no_op():
    image, target = _sample()
    out_image, out_target = RandomHorizontalFlip(p=0.0)(image, target)

    assert torch.equal(out_image, image)
    assert torch.equal(out_target["boxes"], target["boxes"])


def test_flip_does_not_mutate_the_caller_target():
    """The dataset reuses its annotation dicts, so in-place edits would corrupt them."""
    image, target = _sample()
    original = target["boxes"].clone()
    RandomHorizontalFlip(p=1.0)(image, target)
    assert torch.equal(target["boxes"], original)


def test_flip_handles_an_image_with_no_objects():
    image = torch.zeros(3, 20, 20)
    target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64),
              "masks": torch.zeros((0, 20, 20), dtype=torch.uint8)}

    out_image, out_target = RandomHorizontalFlip(p=1.0)(image, target)
    assert out_target["boxes"].shape == (0, 4)
    assert out_image.shape == (3, 20, 20)


def test_flip_works_without_a_masks_key():
    """The car-parts detector trains on boxes only."""
    image, target = _sample()
    del target["masks"]
    _, out = RandomHorizontalFlip(p=1.0)(image, target)
    assert "masks" not in out
    assert out["boxes"].tolist() == [[70.0, 20.0, 90.0, 40.0]]


def test_jitter_changes_pixels_but_never_the_target():
    image, target = _sample()
    out_image, out_target = RandomPhotometricJitter(p=1.0)(image, target)

    assert torch.equal(out_target["boxes"], target["boxes"])
    assert torch.equal(out_target["masks"], target["masks"])
    assert out_image.shape == image.shape


def test_jitter_keeps_values_in_the_unit_range():
    # A mid-gray image is the case where brightness can push past 1.0.
    image = torch.full((3, 20, 20), 0.9)
    target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)}

    for seed in range(20):
        random.seed(seed)
        torch.manual_seed(seed)
        out, _ = RandomPhotometricJitter(p=1.0)(image.clone(), target)
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_jitter_with_zero_probability_is_a_no_op():
    image, target = _sample()
    out_image, _ = RandomPhotometricJitter(p=0.0)(image, target)
    assert torch.equal(out_image, image)


def test_compose_applies_transforms_in_order():
    calls = []

    def first(image, target):
        calls.append("first")
        return image, target

    def second(image, target):
        calls.append("second")
        return image, target

    Compose([first, second])(*_sample())
    assert calls == ["first", "second"]


def test_build_train_transforms_returns_none_when_everything_is_disabled():
    assert build_train_transforms(hflip_prob=0.0, jitter_prob=0.0) is None


def test_build_train_transforms_default_pipeline_runs():
    image, target = _sample()
    out_image, out_target = build_train_transforms()(image, target)
    assert out_image.shape == image.shape
    assert out_target["boxes"].shape == target["boxes"].shape


@pytest.mark.parametrize("cfg", [CAR_PARTS, DAMAGE], ids=["car_parts", "damage"])
def test_flip_on_a_real_annotated_sample_keeps_box_around_mask(cfg):
    """End-to-end on real data: after a flip the box must still bound the mask."""
    try:
        dataset = build_dataset(cfg, cfg.train_split, transforms=RandomHorizontalFlip(p=1.0))
    except FileNotFoundError:
        pytest.skip(f"{cfg.name} dataset not present under data/raw/")

    image, target = dataset[0]
    assert image.shape[1:] == (image.shape[1], image.shape[2])

    boxes = target["boxes"]
    assert torch.all(boxes[:, 2] > boxes[:, 0]), "flip produced inverted boxes"
    assert torch.all(boxes[:, 0] >= 0)
    assert torch.all(boxes[:, 2] <= image.shape[2] + 1)

    if not cfg.with_masks:
        return

    for box, mask in zip(boxes, target["masks"]):
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if not xs.numel():
            continue
        # Allow a pixel of slack: boxes are floats, the mask is rasterized.
        assert xs.min().item() >= int(box[0]) - 2
        assert xs.max().item() <= int(box[2]) + 2
