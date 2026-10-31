"""Same-view check between A and B, on synthetic scenes."""

import cv2
import numpy as np

from src.m4_comparison.view_check import car_region, check_same_view

W, H = 640, 480


def texture(seed, width=W, height=H):
    """Blocks of random color, rich in SIFT keypoints."""
    blocks = np.random.default_rng(seed).integers(0, 255, (height // 8, width // 8, 3), dtype=np.uint8)
    return cv2.resize(blocks, (width, height), interpolation=cv2.INTER_NEAREST)


def shifted(image, dx, dy):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)


def full_region():
    return np.ones((H, W), dtype=bool)


def test_same_view_with_a_small_shift_is_comparable():
    image = texture(1)
    result = check_same_view(image, shifted(image, 15, 10), full_region(), full_region())
    assert result.comparable is True
    assert result.puntos >= result.umbral_puntos and result.solape >= result.umbral_solape


def test_unrelated_photos_are_not_comparable():
    result = check_same_view(texture(1), texture(2), full_region(), full_region())
    assert result.comparable is False
    assert result.puntos < result.umbral_puntos


def test_shared_background_does_not_count():
    background = texture(3)
    image_a, image_b = background.copy(), background.copy()
    image_a[100:380, 160:480] = texture(4)[100:380, 160:480]
    image_b[100:380, 160:480] = texture(5)[100:380, 160:480]
    car = np.zeros((H, W), dtype=bool)
    car[100:380, 160:480] = True

    assert check_same_view(image_a, image_b, full_region(), full_region()).puntos > 40  # background alone
    assert check_same_view(image_a, image_b, car, car).comparable is False


def test_two_shots_that_only_share_a_panel_fail_on_overlap():
    scene = texture(6, width=W + 480)
    first, second = scene[:, :W], scene[:, 480:]  # share a strip 160 px wide
    result = check_same_view(first, second, full_region(), full_region(), min_points=8)
    assert result.puntos >= 8
    assert result.solape < result.umbral_solape and result.comparable is False


def test_photo_without_parts_is_not_comparable():
    image = texture(7)
    empty = car_region(np.zeros((0, H, W), dtype=bool), H, W)
    result = check_same_view(image, image, empty, empty)
    assert result.comparable is False and result.puntos == 0 and result.solape == 0.0


def test_flat_image_does_not_fail():
    flat = np.full((H, W, 3), 128, dtype=np.uint8)
    assert check_same_view(flat, flat, full_region(), full_region()).comparable is False


def test_large_photo_and_different_sizes_are_handled():
    small = texture(8)
    large = cv2.resize(small, (W * 3, H * 3), interpolation=cv2.INTER_NEAREST)
    region = np.ones(large.shape[:2], dtype=bool)
    assert check_same_view(small, large, full_region(), region).comparable is True


def test_car_region_is_the_union_of_the_parts():
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0, 0] = masks[1, 3, 3] = True
    assert car_region(masks, 4, 4).sum() == 2


def test_report_json_has_the_thresholds():
    image = texture(9)
    record = check_same_view(image, image, full_region(), full_region(), min_points=10, min_overlap=0.2).to_json()
    assert set(record) == {"comparable", "puntos", "solape", "umbral_puntos", "umbral_solape"}
    assert record["umbral_puntos"] == 10 and record["umbral_solape"] == 0.2
