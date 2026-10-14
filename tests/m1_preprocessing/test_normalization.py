import cv2
import numpy as np
import pytest

from src.m1_preprocessing.normalization import (apply_clahe_rgb, preprocess_pipeline,
                                                resize_with_padding, to_pytorch_tensor_format)


def _gradient(height=96, width=128):
    """Low-contrast horizontal ramp, gray: CLAHE should spread it out."""
    ramp = np.linspace(100, 140, width, dtype=np.uint8)
    return np.repeat(np.tile(ramp, (height, 1))[:, :, None], 3, axis=2)


def test_clahe_keeps_shape_and_dtype_and_raises_contrast():
    image = _gradient()
    out = apply_clahe_rgb(image)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert out.std() > image.std()


def test_clahe_leaves_the_input_untouched_and_gray_stays_gray():
    image = _gradient()
    before = image.copy()
    out = apply_clahe_rgb(image)
    assert np.array_equal(image, before)
    # Only L changes: a neutral image must not pick up a color cast.
    assert np.abs(out[..., 0].astype(int) - out[..., 2].astype(int)).max() <= 2


def test_clahe_clip_limit_changes_the_result():
    image = _gradient()
    assert not np.array_equal(apply_clahe_rgb(image, clip_limit=1.0),
                              apply_clahe_rgb(image, clip_limit=4.0))


@pytest.mark.parametrize("height, width", [(100, 200), (200, 100), (512, 512), (30, 40)])
def test_resize_with_padding_returns_the_target_size(height, width):
    out = resize_with_padding(np.full((height, width, 3), 255, np.uint8), target_size=(512, 512))
    assert out.shape == (512, 512, 3) and out.dtype == np.uint8


def test_resize_with_padding_centers_the_image_and_pads_with_black():
    out = resize_with_padding(np.full((100, 200, 3), 255, np.uint8), target_size=(512, 512))
    # 200x100 scales to 512x256, leaving 128 black rows above and below.
    assert out[:128].max() == 0 and out[384:].max() == 0
    assert out[128:384].min() == 255


def test_tensor_format_is_chw_rgb_and_imagenet_normalized():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[..., 2] = 255  # pure red in BGR
    tensor = to_pytorch_tensor_format(image)
    assert tensor.shape == (3, 4, 6) and tensor.dtype == np.float32
    expected = (np.array([1.0, 0.0, 0.0]) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    assert np.allclose(tensor[:, 0, 0], expected, atol=1e-5)


def test_legacy_pipeline_output_shape(tmp_path):
    path = str(tmp_path / "img.png")
    cv2.imwrite(path, _gradient())
    assert preprocess_pipeline(path).shape == (3, 512, 512)


def test_legacy_pipeline_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess_pipeline(str(tmp_path / "missing.png"))
