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
def test_jpg_and_png_at_minimum_size_pass(tmp_path, fmt, name, expected):
    info = validate_image(_save(tmp_path / name, 1067, 800, fmt))
    assert info == ImageInfo(path=str(tmp_path / name), format=expected, width=1067, height=800,
                             exif_orientation=1)


@pytest.mark.parametrize("width, height", [(800, 800), (896, 1195), (1216, 874), (1920, 1080)])
def test_photos_below_full_hd_pass_when_short_side_reaches_800(tmp_path, width, height):
    # AI-edited and messaging-app photos of the own batches (896x1195, 1216x874) are accepted.
    info = validate_image(_save(tmp_path / "ok.jpg", width, height, "JPEG"))
    assert (info.width, info.height) == (width, height)


def test_portrait_photo_passes_like_landscape(tmp_path):
    info = validate_image(_save(tmp_path / "p.jpg", 800, 1067, "JPEG"))
    assert (info.width, info.height) == (800, 1067)


@pytest.mark.parametrize("width, height", [(1067, 799), (799, 4000), (640, 480), (4000, 600)])
def test_short_side_under_800_is_rejected(tmp_path, width, height):
    with pytest.raises(ImageValidationError) as error:
        validate_image(_save(tmp_path / "small.jpg", width, height, "JPEG"))
    assert error.value.code == "resolucion"
    assert f"{width}×{height}" in str(error.value) and "800 píxeles" in str(error.value)


def test_minimum_size_is_configurable(tmp_path):
    path = _save(tmp_path / "small.png", 640, 640, "PNG")
    assert validate_image(path, min_short_side=640).width == 640
    with pytest.raises(ImageValidationError):
        validate_image(path, min_short_side=641)


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
