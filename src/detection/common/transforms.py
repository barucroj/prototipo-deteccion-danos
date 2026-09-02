"""Training-time data augmentation for both detectors.

Only applied to the training split. Never augment validation or test: the
metric has to be measured on the real images, otherwise scores stop being
comparable across runs.

Every transform here rewrites the target alongside the image. That coupling is
the whole risk of augmentation for detection: if a flipped image keeps
un-flipped boxes, training silently optimizes against wrong labels and the
only symptom is a model that quietly gets worse. The flip is therefore covered
by tests that check boxes and masks still land on the object.

Which transforms, and why these:

- Horizontal flip — a mirrored car is still a valid car, so this roughly
  doubles the effective dataset for free. The single highest-value transform
  here, especially for the 333-image car-parts split.
- Brightness / contrast / saturation jitter — rental photos are taken by
  non-experts under whatever lighting exists, so this is the project's real
  domain variation rather than a generic trick.

Deliberately excluded: vertical flips and large rotations (an upside-down car
does not occur in this domain and teaches the model noise), and random crops
(a crop can cut a scratch in half and leave an annotation that no longer
describes what is in the image).
"""

from __future__ import annotations

import random

import torch
from torchvision.transforms import ColorJitter


class Compose:
    """Chains transforms that each take and return ``(image, target)``."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def __repr__(self):
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{inner}])"


class RandomHorizontalFlip:
    """Mirrors the image and its boxes and masks with probability ``p``.

    Boxes are xyxy in pixels, so a mirror maps ``[x1, x2]`` to
    ``[W - x2, W - x1]`` — the coordinates swap places, which is why this
    cannot be written as a single subtraction per corner.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        width = image.shape[-1]
        image = image.flip(-1)

        target = dict(target)
        boxes = target["boxes"]
        if boxes.numel():
            flipped = boxes.clone()
            flipped[:, 0] = width - boxes[:, 2]
            flipped[:, 2] = width - boxes[:, 0]
            target["boxes"] = flipped

        if "masks" in target and target["masks"].numel():
            target["masks"] = target["masks"].flip(-1)

        return image, target

    def __repr__(self):
        return f"RandomHorizontalFlip(p={self.p})"


class RandomPhotometricJitter:
    """Randomly perturbs brightness, contrast and saturation.

    Geometry is untouched, so the target passes through unchanged. Hue is
    deliberately left alone: recoloring a car does not reflect real capture
    variation and would only blur the color cues that separate a rust-stained
    scratch from a shadow.
    """

    def __init__(self, brightness: float = 0.3, contrast: float = 0.3,
                 saturation: float = 0.2, p: float = 0.5):
        self.p = p
        self.jitter = ColorJitter(brightness=brightness, contrast=contrast,
                                  saturation=saturation)

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        # ColorJitter can push values outside [0, 1]; detection backbones
        # expect the same range the dataset produces.
        return self.jitter(image).clamp_(0.0, 1.0), target

    def __repr__(self):
        return f"RandomPhotometricJitter(p={self.p})"


def build_train_transforms(hflip_prob: float = 0.5, jitter_prob: float = 0.5,
                           brightness: float = 0.3, contrast: float = 0.3,
                           saturation: float = 0.2):
    """The default training augmentation pipeline for both detectors.

    Returns ``None`` when every probability is zero, so callers can pass the
    result straight to the dataset to mean "no augmentation".
    """
    transforms = []
    if hflip_prob > 0:
        transforms.append(RandomHorizontalFlip(hflip_prob))
    if jitter_prob > 0:
        transforms.append(RandomPhotometricJitter(brightness, contrast,
                                                  saturation, jitter_prob))
    return Compose(transforms) if transforms else None
