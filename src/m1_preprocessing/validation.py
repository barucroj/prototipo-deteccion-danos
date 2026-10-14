"""Input validation for the photos of a rental pair (thesis module M1, RF02).

Checked before anything is processed, so a bad file is reported to the user
instead of failing somewhere inside a detector:

* the file is a real JPG or PNG — decided from its content, not its extension;
* it has at least the protocol's resolution, 1920x1080.

The resolution rule is orientation-free: the long side must reach 1920 and the
short side 1080, so a portrait 1080x1920 photo passes like a landscape one and
the result does not depend on whether the EXIF rotation was applied yet.

This is the system's entry check, on purpose kept out of
:func:`src.m1_preprocessing.pipeline.preprocess_image`: the training datasets
are 640x640 and ~1000 px images and must still go through the optical steps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

#: Pillow format name -> the name shown to the user.
ALLOWED_FORMATS = {"JPEG": "JPG", "PNG": "PNG"}
MIN_LONG_SIDE = 1920
MIN_SHORT_SIDE = 1080

_EXIF_ORIENTATION_TAG = 0x0112


class ImageValidationError(ValueError):
    """A photo that cannot be analyzed. ``str(error)`` is the message for the user.

    Args:
        code: Machine-readable reason: ``"no_encontrado"``, ``"ilegible"``,
            ``"formato"`` or ``"resolucion"``.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ImageInfo:
    """What the file says about itself, before any transformation.

    Args:
        path: The file that was inspected.
        format: ``"JPG"`` or ``"PNG"``.
        width / height: Stored pixel size, i.e. *before* the EXIF rotation.
        exif_orientation: EXIF orientation tag, 1-8; 1 (upright) when absent.
    """

    path: str
    format: str
    width: int
    height: int
    exif_orientation: int


def inspect_image(path: str) -> ImageInfo:
    """Read format, size and EXIF orientation; reject unreadable or non JPG/PNG files."""
    name = os.path.basename(path)
    if not os.path.isfile(path):
        raise ImageValidationError("no_encontrado", f"No se encontró el archivo: {path}")

    try:
        with Image.open(path) as image:
            image.verify()
        # verify() leaves the file object unusable, so the header is read again.
        with Image.open(path) as image:
            pil_format = image.format
            width, height = image.size
            orientation = int(image.getexif().get(_EXIF_ORIENTATION_TAG, 1) or 1)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise ImageValidationError(
            "ilegible", f"{name}: el archivo no es una imagen válida o está dañado.") from error

    if pil_format not in ALLOWED_FORMATS:
        raise ImageValidationError(
            "formato",
            f"{name}: formato {pil_format or 'desconocido'} no admitido; "
            f"se aceptan {' y '.join(sorted(ALLOWED_FORMATS.values()))}.")

    if not 1 <= orientation <= 8:
        orientation = 1
    return ImageInfo(path=path, format=ALLOWED_FORMATS[pil_format], width=width, height=height,
                     exif_orientation=orientation)


def validate_image(path: str, min_long_side: int = MIN_LONG_SIDE,
                   min_short_side: int = MIN_SHORT_SIDE) -> ImageInfo:
    """:func:`inspect_image` plus the minimum-resolution check.

    Raises:
        ImageValidationError: with ``code`` set to the first rule that failed.
    """
    info = inspect_image(path)
    long_side, short_side = max(info.width, info.height), min(info.width, info.height)
    if long_side < min_long_side or short_side < min_short_side:
        raise ImageValidationError(
            "resolucion",
            f"{os.path.basename(path)}: resolución {info.width}×{info.height} insuficiente; "
            f"el mínimo es {min_long_side}×{min_short_side} píxeles.")
    return info
