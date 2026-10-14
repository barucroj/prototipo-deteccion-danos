import json

import cv2
import numpy as np
import pytest
from PIL import Image

from src.m1_preprocessing.pipeline import (PreprocessConfig, load_image, preprocess_file,
                                           preprocess_image)
from src.m1_preprocessing.specular_removal import ShenParams, remove_specular_highlights
from src.m1_preprocessing.validation import ImageValidationError

OFF = PreprocessConfig(specular=False, clahe=False)


def _car_like(height=120, width=160):
    """Red body with a white glare patch: the highlight removal has work to do."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (30, 30, 170)  # BGR
    image[40:70, 60:100] = (250, 250, 255)
    return image


def _save_png(path, image_bgr, orientation=None):
    kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kwargs["exif"] = exif
    Image.fromarray(image_bgr[:, :, ::-1]).save(path, format="PNG", **kwargs)
    return str(path)


# ---------------------------------------------------------------- preprocess_image
def test_defaults_are_specular_on_clahe_off():
    config = PreprocessConfig()
    assert config.specular is True and config.clahe is False


def test_all_steps_off_returns_the_input_unchanged():
    image = _car_like()
    out, steps = preprocess_image(image, OFF)
    assert steps == [] and np.array_equal(out, image)


def test_output_keeps_size_and_dtype_for_every_combination():
    image = _car_like()
    for specular in (False, True):
        for clahe in (False, True):
            out, _ = preprocess_image(image, PreprocessConfig(specular=specular, clahe=clahe))
            assert out.shape == image.shape and out.dtype == np.uint8


def test_default_config_applies_only_the_highlight_removal():
    image = _car_like()
    out, steps = preprocess_image(image)
    assert [s["nombre"] for s in steps] == ["reflejos_especulares"]
    assert np.array_equal(out, remove_specular_highlights(image, ShenParams()))
    assert out[40:70, 60:100].mean() < image[40:70, 60:100].mean()


def test_steps_run_specular_first_then_clahe_and_record_their_parameters():
    config = PreprocessConfig(specular=True, clahe=True, shen=ShenParams(brightness_threshold=150.0),
                              clahe_clip_limit=3.0, clahe_tile_grid=(4, 4))
    _, steps = preprocess_image(_car_like(), config)
    assert [s["nombre"] for s in steps] == ["reflejos_especulares", "clahe"]
    assert steps[0]["parametros"]["brightness_threshold"] == 150.0
    assert steps[1]["parametros"] == {"clip_limit": 3.0, "tile_grid_size": [4, 4]}


def test_input_is_not_modified():
    image = _car_like()
    before = image.copy()
    preprocess_image(image, PreprocessConfig(specular=True, clahe=True))
    assert np.array_equal(image, before)


@pytest.mark.parametrize("bad", [np.zeros((10, 10), np.uint8), np.zeros((10, 10, 4), np.uint8),
                                 np.zeros((10, 10, 3), np.float32)])
def test_wrong_image_layout_raises(bad):
    with pytest.raises(ValueError):
        preprocess_image(bad)


# ---------------------------------------------------------------------- load_image
def test_load_image_returns_bgr(tmp_path):
    image = _car_like()
    loaded, info = load_image(_save_png(tmp_path / "car.png", image))
    assert np.array_equal(loaded, image)
    assert info.format == "PNG" and info.exif_orientation == 1


def test_load_image_matches_cv2_on_a_jpg(tmp_path):
    path = str(tmp_path / "car.jpg")
    cv2.imwrite(path, _car_like())
    loaded, _ = load_image(path)
    # Two JPEG decoders: allow rounding differences, not a channel swap.
    assert np.abs(loaded.astype(int) - cv2.imread(path).astype(int)).max() <= 3


def test_load_image_applies_exif_orientation(tmp_path):
    image = _car_like(height=40, width=100)
    image[0:5, 0:5] = (0, 255, 0)  # marker in the stored top-left corner
    # Orientation 6: the stored image must be rotated 90 degrees clockwise.
    loaded, info = load_image(_save_png(tmp_path / "rot.png", image, orientation=6))
    assert info.exif_orientation == 6 and (info.width, info.height) == (100, 40)
    assert loaded.shape[:2] == (100, 40)
    assert np.array_equal(loaded, np.rot90(image, k=-1))
    assert tuple(loaded[2, 37]) == (0, 255, 0)  # the marker is now top-right


def test_load_image_drops_alpha(tmp_path):
    rgba = np.zeros((20, 30, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 3] = 200, 128
    Image.fromarray(rgba).save(tmp_path / "alpha.png")
    loaded, _ = load_image(str(tmp_path / "alpha.png"))
    assert loaded.shape == (20, 30, 3) and tuple(loaded[0, 0]) == (0, 0, 200)


# ----------------------------------------------------------------- preprocess_file
def test_preprocess_file_rejects_low_resolution_by_default(tmp_path):
    with pytest.raises(ImageValidationError) as error:
        preprocess_file(_save_png(tmp_path / "small.png", _car_like()))
    assert error.value.code == "resolucion"


def test_preprocess_file_metadata(tmp_path):
    image = cv2.resize(_car_like(), (1920, 1080), interpolation=cv2.INTER_NEAREST)
    result = preprocess_file(_save_png(tmp_path / "car.png", image))

    assert result.image.shape == (1080, 1920, 3) and result.image.dtype == np.uint8
    meta = result.metadata
    assert meta["archivo"] == "car.png" and meta["formato"] == "PNG"
    assert (meta["width_original"], meta["height_original"]) == (1920, 1080)
    assert (meta["width"], meta["height"]) == (1920, 1080)
    assert meta["orientacion_exif"] == 1
    assert [s["nombre"] for s in meta["transformaciones"]] == ["reflejos_especulares"]
    json.dumps(meta)  # must be serializable as is


def test_preprocess_file_records_the_exif_rotation_and_the_new_size(tmp_path):
    image = _car_like(height=40, width=100)
    result = preprocess_file(_save_png(tmp_path / "rot.png", image, orientation=8), OFF,
                             validate=False)
    meta = result.metadata
    assert (meta["width_original"], meta["height_original"]) == (100, 40)
    assert (meta["width"], meta["height"]) == (40, 100)
    assert meta["transformaciones"] == [{"nombre": "orientacion_exif", "parametros": {"orientacion": 8}}]
    assert np.array_equal(result.image, np.rot90(image, k=1))


def test_preprocess_file_without_validation_still_checks_the_format(tmp_path):
    path = tmp_path / "car.bmp"
    Image.fromarray(_car_like()).save(path, format="BMP")
    with pytest.raises(ImageValidationError) as error:
        preprocess_file(str(path), validate=False)
    assert error.value.code == "formato"
