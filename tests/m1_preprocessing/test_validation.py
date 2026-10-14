import numpy as np
import pytest
from PIL import Image

from src.m1_preprocessing.validation import (ImageInfo, ImageValidationError, inspect_image,
                                             validate_image)


def _save(path, width, height, fmt, orientation=None):
    image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kwargs["exif"] = exif
    image.save(path, format=fmt, **kwargs)
    return str(path)


@pytest.mark.parametrize("fmt, name, expected", [("JPEG", "a.jpg", "JPG"), ("PNG", "a.png", "PNG")])
def test_jpg_and_png_at_minimum_resolution_pass(tmp_path, fmt, name, expected):
    info = validate_image(_save(tmp_path / name, 1920, 1080, fmt))
    assert info == ImageInfo(path=str(tmp_path / name), format=expected, width=1920, height=1080,
                             exif_orientation=1)


def test_portrait_photo_passes_like_landscape(tmp_path):
    info = validate_image(_save(tmp_path / "p.jpg", 1080, 1920, "JPEG"))
    assert (info.width, info.height) == (1080, 1920)


@pytest.mark.parametrize("width, height", [(1919, 1080), (1920, 1079), (1280, 960), (1079, 4000)])
def test_low_resolution_is_rejected(tmp_path, width, height):
    with pytest.raises(ImageValidationError) as error:
        validate_image(_save(tmp_path / "small.jpg", width, height, "JPEG"))
    assert error.value.code == "resolucion"
    assert f"{width}×{height}" in str(error.value) and "1920×1080" in str(error.value)


def test_minimum_resolution_is_configurable(tmp_path):
    path = _save(tmp_path / "small.png", 640, 640, "PNG")
    assert validate_image(path, min_long_side=640, min_short_side=640).width == 640


def test_format_is_decided_by_content_not_extension(tmp_path):
    # A BMP named .jpg is rejected; a PNG named .jpg is accepted as PNG.
    with pytest.raises(ImageValidationError) as error:
        validate_image(_save(tmp_path / "fake.jpg", 1920, 1080, "BMP"))
    assert error.value.code == "formato" and "BMP" in str(error.value)

    assert validate_image(_save(tmp_path / "really_png.jpg", 1920, 1080, "PNG")).format == "PNG"


def test_non_image_file_is_rejected(tmp_path):
    path = tmp_path / "notes.jpg"
    path.write_text("this is not an image")
    with pytest.raises(ImageValidationError) as error:
        validate_image(str(path))
    assert error.value.code == "ilegible"


def test_truncated_file_is_rejected(tmp_path):
    path = _save(tmp_path / "cut.png", 1920, 1080, "PNG")
    data = open(path, "rb").read()
    open(path, "wb").write(data[:len(data) // 2])
    with pytest.raises(ImageValidationError) as error:
        validate_image(path)
    assert error.value.code == "ilegible"


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ImageValidationError) as error:
        validate_image(str(tmp_path / "nope.jpg"))
    assert error.value.code == "no_encontrado"


def test_exif_orientation_is_reported_and_size_is_the_stored_one(tmp_path):
    info = inspect_image(_save(tmp_path / "rotated.jpg", 1920, 1080, "JPEG", orientation=6))
    assert info.exif_orientation == 6
    assert (info.width, info.height) == (1920, 1080)


def test_inspect_does_not_check_resolution(tmp_path):
    assert inspect_image(_save(tmp_path / "tiny.png", 32, 16, "PNG")).width == 32
