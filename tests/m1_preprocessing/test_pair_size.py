import cv2
import numpy as np
import pytest
from PIL import Image

from src.m1_preprocessing.pair_size import STEP_NAME, limit_long_side, pair_max_long_sides, upright_size
from src.m1_preprocessing.pipeline import PreprocessConfig, preprocess_file
from src.m1_preprocessing.validation import ImageInfo

OFF = PreprocessConfig(specular=False, clahe=False)


def test_only_the_larger_photo_gets_a_limit():
    assert pair_max_long_sides((1024, 1024), (2048, 2048)) == (None, 1024)
    assert pair_max_long_sides((3464, 3464), (1170, 1170)) == (1170, None)
    assert pair_max_long_sides((1536, 1024), (1536, 1024)) == (None, None)
    # different aspect ratio: long sides are matched
    assert pair_max_long_sides((4000, 3000), (1080, 1920)) == (1920, None)


def test_upright_size_swaps_on_rotated_exif():
    assert upright_size(ImageInfo("x", "JPG", 4000, 3000, 1)) == (4000, 3000)
    assert upright_size(ImageInfo("x", "JPG", 4000, 3000, 6)) == (3000, 4000)


def test_limit_long_side_reduces_and_keeps_aspect():
    image = np.zeros((1792, 2392, 3), np.uint8)
    reduced, step = limit_long_side(image, 1196)
    assert reduced.shape == (896, 1196, 3)
    assert step["nombre"] == STEP_NAME
    assert step["parametros"]["de"] == [2392, 1792] and step["parametros"]["a"] == [1196, 896]
    assert step["parametros"]["escala"] == 0.5


def test_limit_long_side_never_enlarges():
    image = np.zeros((100, 200, 3), np.uint8)
    for limit in (None, 200, 500):
        same, step = limit_long_side(image, limit)
        assert same is image and step is None
    with pytest.raises(ValueError):
        limit_long_side(image, 0)


def test_preprocess_file_reduces_after_exif_and_records_the_step(tmp_path):
    path = tmp_path / "rot.jpg"
    image = Image.fromarray(np.full((300, 400, 3), 128, np.uint8))  # stored 400x300
    exif = image.getexif()
    exif[0x0112] = 6  # rotated: upright 300x400
    image.save(path, exif=exif)

    result = preprocess_file(str(path), OFF, validate=False, max_long_side=200)
    assert result.image.shape[:2] == (200, 150)
    assert (result.metadata["width"], result.metadata["height"]) == (150, 200)
    names = [t["nombre"] for t in result.metadata["transformaciones"]]
    assert names == ["orientacion_exif", STEP_NAME]

    untouched = preprocess_file(str(path), OFF, validate=False)
    assert untouched.image.shape[:2] == (400, 300)


def test_reduced_image_matches_area_interpolation(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (400, 600, 3), dtype=np.uint8)
    reduced, _ = limit_long_side(image, 300)
    expected = cv2.resize(image, (300, 200), interpolation=cv2.INTER_AREA)
    assert np.array_equal(reduced, expected)
