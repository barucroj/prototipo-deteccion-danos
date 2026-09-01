import glob
import os

import cv2
import numpy as np
import pytest

from src.preprocessing.specular_removal import ShenParams, remove_specular_highlights

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Candidate folders to draw the 10 sample images from, tried in order.
SAMPLE_GLOBS = [
    os.path.join(PROJECT_ROOT, "data", "raw", "CarDD_release", "CarDD_release", "CarDD_COCO", "train2017", "*.jpg"),
    os.path.join(PROJECT_ROOT, "data", "raw", "Car parts coco-segmentation", "train", "*.jpg"),
    os.path.join(PROJECT_ROOT, "data", "raw", "Car-Parts-Segmentation", "trainingset", "JPEGImages", "*.jpg"),
]


def _find_sample_images(n=10):
    for pattern in SAMPLE_GLOBS:
        files = sorted(glob.glob(pattern))
        if len(files) >= n:
            return files[:n]
    return []


SAMPLE_IMAGES = _find_sample_images(10)


@pytest.mark.skipif(not SAMPLE_IMAGES, reason="No sample images found under data/raw")
@pytest.mark.parametrize("image_path", SAMPLE_IMAGES, ids=[os.path.basename(p) for p in SAMPLE_IMAGES])
def test_remove_specular_highlights_on_sample_images(image_path):
    image = cv2.imread(image_path)
    assert image is not None, f"Could not read {image_path}"

    result = remove_specular_highlights(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0 and result.max() <= 255
    # The diffuse-only output should never be brighter than the original
    # per channel (small tolerance for rounding).
    assert (result.astype(np.int16) <= image.astype(np.int16) + 1).all()


def test_remove_specular_highlights_reduces_synthetic_highlight():
    # Uniform diffuse red background with a bright achromatic (white)
    # highlight spot in the middle.
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 2] = 120  # BGR order: index 2 is R.
    image[40:60, 40:60] = [255, 255, 255]

    result = remove_specular_highlights(image)

    highlight_before = int(image[50, 50].astype(np.int16).sum())
    highlight_after = int(result[50, 50].astype(np.int16).sum())
    background_before = int(image[10, 10].astype(np.int16).sum())
    background_after = int(result[10, 10].astype(np.int16).sum())

    assert highlight_after < highlight_before
    # Background pixels (already diffuse, no highlight) should be
    # essentially unchanged.
    assert abs(background_after - background_before) < 15


def test_remove_specular_highlights_invalid_shape_raises():
    with pytest.raises(ValueError):
        remove_specular_highlights(np.zeros((10, 10), dtype=np.uint8))


def test_shen_params_defaults_are_sane():
    params = ShenParams()
    assert params.epsilon > 0
    assert params.smooth_ksize >= 0
    assert 1 / 3 < params.diffuse_chromaticity_clip < 1


def test_remove_specular_highlights_custom_params():
    image = np.full((20, 20, 3), 200, dtype=np.uint8)
    params = ShenParams(epsilon=1e-3, smooth_ksize=0, diffuse_chromaticity_clip=0.4)

    result = remove_specular_highlights(image, params=params)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
