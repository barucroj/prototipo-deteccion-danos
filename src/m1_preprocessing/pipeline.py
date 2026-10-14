"""The M1 optical preprocessing pipeline (thesis module M1, RF03).

Turns a photo file into the image the detectors receive, plus the metadata
that makes the result traceable:

1. validate format and resolution (:mod:`src.m1_preprocessing.validation`);
2. load it upright, applying the EXIF orientation;
3. reduce specular highlights (on by default);
4. enhance contrast with CLAHE (off by default).

The image keeps its size and stays 8-bit BGR. There is **no resize and no
ImageNet normalization** here: torchvision's R-CNNs take RGB in [0, 1] at the
original size and do both inside ``GeneralizedRCNNTransform``.
``normalization.preprocess_pipeline`` (512x512 + ImageNet) is not part of this
path.

Steps 3 and 4 are photometric, so boxes and masks computed on the output are
valid on the input and vice versa.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from src.m1_preprocessing.normalization import apply_clahe_rgb
from src.m1_preprocessing.specular_removal import ShenParams, remove_specular_highlights
from src.m1_preprocessing.validation import (MIN_LONG_SIDE, MIN_SHORT_SIDE, ImageInfo,
                                             inspect_image, validate_image)


@dataclass(frozen=True)
class PreprocessConfig:
    """Which optical steps run, and with what parameters.

    Args:
        specular: Reduce specular highlights (Shen & Cai, see
            :mod:`src.m1_preprocessing.specular_removal`).
        clahe: Apply CLAHE on the L channel. Optional in the spec, so off.
        shen: Parameters of the highlight removal.
        clahe_clip_limit / clahe_tile_grid: Parameters of CLAHE.
    """

    specular: bool = True
    clahe: bool = False
    shen: ShenParams = field(default_factory=ShenParams)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)


@dataclass
class PreprocessResult:
    """Output of M1 for one photo.

    Args:
        image: (H, W, 3) uint8 BGR image, ready for the detectors.
        metadata: JSON-serializable record of the source file and of every
            transformation applied, in order (see :func:`preprocess_file`).
    """

    image: np.ndarray
    metadata: dict


def load_image(path: str) -> Tuple[np.ndarray, ImageInfo]:
    """Load a JPG/PNG as upright 8-bit BGR, applying its EXIF orientation.

    ``cv2.imread`` also honors EXIF but ``PIL.Image.open`` does not; loading
    every photo through here keeps the two code paths from disagreeing.
    Alpha is dropped.

    Raises:
        ImageValidationError: if the file is missing, unreadable or not JPG/PNG.
    """
    info = inspect_image(path)
    with Image.open(path) as image:
        upright = ImageOps.exif_transpose(image).convert("RGB")
    return np.ascontiguousarray(np.asarray(upright)[:, :, ::-1]), info


def preprocess_image(image_bgr: np.ndarray,
                     config: Optional[PreprocessConfig] = None) -> Tuple[np.ndarray, List[dict]]:
    """Run the optical steps on an image already in memory.

    Highlights go first: CLAHE would otherwise stretch contrast around the
    glare that the next step is about to remove.

    Returns:
        ``(image, steps)``: the processed (H, W, 3) uint8 BGR image, same
        size as the input, and one ``{"nombre", "parametros"}`` dict per step
        applied. With every step off the input is returned unchanged.
    """
    config = config or PreprocessConfig()
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError(
            f"expected an (H, W, 3) uint8 BGR image, got {image_bgr.shape} {image_bgr.dtype}")

    steps: List[dict] = []
    image = image_bgr
    if config.specular:
        image = remove_specular_highlights(image, config.shen)
        steps.append({"nombre": "reflejos_especulares", "parametros": asdict(config.shen)})
    if config.clahe:
        image = apply_clahe_rgb(image, clip_limit=config.clahe_clip_limit,
                                tile_grid_size=tuple(config.clahe_tile_grid))
        steps.append({"nombre": "clahe",
                      "parametros": {"clip_limit": config.clahe_clip_limit,
                                     "tile_grid_size": list(config.clahe_tile_grid)}})
    return image, steps


def preprocess_file(path: str, config: Optional[PreprocessConfig] = None, validate: bool = True,
                    min_long_side: int = MIN_LONG_SIDE,
                    min_short_side: int = MIN_SHORT_SIDE) -> PreprocessResult:
    """M1 end to end for one photo file.

    Args:
        path: JPG or PNG file.
        config: Optical steps to run; defaults to :class:`PreprocessConfig`.
        validate: Enforce the minimum resolution. The format is always
            checked. Turn off only for dataset images, which are smaller.

    Returns:
        A :class:`PreprocessResult` whose ``metadata`` is::

            {"archivo": "car.jpg", "formato": "JPG",
             "width_original": 4000, "height_original": 3000,
             "orientacion_exif": 6, "width": 3000, "height": 4000,
             "transformaciones": [{"nombre": "orientacion_exif", "parametros": {...}},
                                  {"nombre": "reflejos_especulares", "parametros": {...}}]}

        ``width_original`` / ``height_original`` are the stored size;
        ``width`` / ``height`` the size of the returned image, which differ
        only when the EXIF orientation is a 90-degree rotation.

    Raises:
        ImageValidationError: when the file fails a check.
    """
    if validate:
        validate_image(path, min_long_side, min_short_side)
    image, info = load_image(path)

    steps: List[dict] = []
    if info.exif_orientation != 1:
        steps.append({"nombre": "orientacion_exif",
                      "parametros": {"orientacion": info.exif_orientation}})
    image, optical_steps = preprocess_image(image, config)
    steps.extend(optical_steps)

    height, width = image.shape[:2]
    metadata = {
        "archivo": os.path.basename(path),
        "formato": info.format,
        "width_original": info.width,
        "height_original": info.height,
        "orientacion_exif": info.exif_orientation,
        "width": int(width),
        "height": int(height),
        "transformaciones": steps,
    }
    return PreprocessResult(image=image, metadata=metadata)
