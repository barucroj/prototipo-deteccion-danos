"""M1 as a dataset transform, for training and evaluating detectors with it.

:mod:`src.m1_preprocessing.pipeline` works on 8-bit BGR arrays; the detection
datasets yield RGB float tensors. :class:`PreprocessTransform` bridges the
two so the same M1 variant can be applied

* at evaluation time (:mod:`src.m1_preprocessing.ap_experiment`), and
* at training time (``--preprocess`` of the shared trainer), to both the
  training and the validation split.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch

from src.m1_preprocessing.pipeline import PreprocessConfig, preprocess_image

BASELINE = "raw"

#: Variant name -> M1 configuration (``None`` = images untouched).
VARIANTS: Dict[str, Optional[PreprocessConfig]] = {
    BASELINE: None,
    "specular": PreprocessConfig(specular=True, clahe=False),
    "clahe": PreprocessConfig(specular=False, clahe=True),
    "both": PreprocessConfig(specular=True, clahe=True),
}


class PreprocessTransform:
    """Dataset transform that applies an M1 variant to the image tensor.

    The dataset yields RGB float tensors in [0, 1]; M1 works on 8-bit BGR.
    The round trip through uint8 is exact for images loaded from 8-bit files.
    Targets pass through untouched: the steps are photometric. Per-image
    processing time is kept in ``seconds``.

    Args:
        config: The M1 steps to run.
        name: Variant name, only used by ``repr`` (it ends up in the logs).
    """

    def __init__(self, config: PreprocessConfig, name: Optional[str] = None):
        self.config = config
        self.name = name
        self.seconds: List[float] = []

    def __call__(self, image: torch.Tensor, target: dict):
        rgb = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        start = time.perf_counter()
        processed, _ = preprocess_image(bgr, self.config)
        self.seconds.append(time.perf_counter() - start)
        out = np.ascontiguousarray(processed[:, :, ::-1]).astype(np.float32) / 255.0
        return torch.from_numpy(out).permute(2, 0, 1), target

    def __repr__(self):
        steps = [step for step, on in (("specular", self.config.specular),
                                       ("clahe", self.config.clahe)) if on]
        label = self.name or "+".join(steps) or "none"
        return f"M1Preprocess({label})"


def build_preprocess_transform(variant: str) -> Optional[PreprocessTransform]:
    """The transform for a variant name, or ``None`` for ``raw``."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown M1 variant {variant!r}; known: {sorted(VARIANTS)}")
    config = VARIANTS[variant]
    return PreprocessTransform(config, variant) if config is not None else None
