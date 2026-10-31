"""Check that photos A and B show the same view of the car (thesis module M4).

M4 compares positions, so two unrelated photos would give a report that looks
valid and is full of false "new" damages. Before comparing, SIFT keypoints are
matched between the photos and a homography is fitted with RANSAC:

* ``puntos`` counts the matches that agree with it;
* ``solape`` is the IoU in B between the car of B and the car of A moved by it,
  which separates the same view from two shots that merely share a panel.

Only keypoints on the car are used (the union of the M2 part masks), because
in a rental lot the background is the same for every car.

This does not prove it is the same vehicle: two identical units could pass.
The thresholds are not calibrated: there are no real A/B pairs yet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

#: Minimum matches consistent with one homography.
DEFAULT_MIN_VIEW_POINTS = 40
#: Minimum IoU between the car regions of A and B.
DEFAULT_MIN_VIEW_OVERLAP = 0.5

#: Longest side used for the check; enough for SIFT and independent of the photo size.
MAX_SIDE = 1000
MAX_KEYPOINTS = 2000
RATIO_TEST = 0.75
RANSAC_PIXELS = 5.0
MIN_MATCHES_FOR_HOMOGRAPHY = 8


@dataclass(frozen=True)
class ViewCheck:
    """Result of the check, stored in the M4 report as ``verificacion_vista``."""

    comparable: bool
    puntos: int
    solape: float
    umbral_puntos: int
    umbral_solape: float

    def to_json(self) -> dict:
        return asdict(self)


def car_region(part_masks: np.ndarray, height: int, width: int) -> np.ndarray:
    """Union of the (N, H, W) part masks; empty when M2 found no part."""
    if len(part_masks) == 0:
        return np.zeros((height, width), dtype=bool)
    return part_masks.any(axis=0)


def _prepare(image_bgr: np.ndarray, region: np.ndarray):
    """Grayscale image and 0/255 mask, with the longest side at ``MAX_SIDE`` at most."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = region.astype(np.uint8) * 255
    scale = MAX_SIDE / max(gray.shape)
    if scale < 1:
        size = (int(round(gray.shape[1] * scale)), int(round(gray.shape[0] * scale)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return gray, mask


def _fit_homography(points_a, descriptors_a, points_b, descriptors_b):
    """Homography from A to B and its number of inliers, ``(None, 0)`` if there is none."""
    if descriptors_a is None or descriptors_b is None or len(points_a) < 2 or len(points_b) < 2:
        return None, 0
    pairs = cv2.BFMatcher().knnMatch(descriptors_a, descriptors_b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]
    if len(good) < MIN_MATCHES_FOR_HOMOGRAPHY:
        return None, 0
    source = np.float32([points_a[m.queryIdx].pt for m in good])
    target = np.float32([points_b[m.trainIdx].pt for m in good])
    homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, RANSAC_PIXELS)
    if homography is None:
        return None, 0
    return homography, int(inliers.sum())


def check_same_view(image_a: np.ndarray, image_b: np.ndarray, region_a: np.ndarray, region_b: np.ndarray,
                    min_points: int = DEFAULT_MIN_VIEW_POINTS,
                    min_overlap: float = DEFAULT_MIN_VIEW_OVERLAP) -> ViewCheck:
    """Decide whether A and B can be compared.

    Args:
        image_a / image_b: BGR images.
        region_a / region_b: Boolean car masks, see :func:`car_region`.
        min_points / min_overlap: Both must be reached.
    """
    gray_a, mask_a = _prepare(image_a, region_a)
    gray_b, mask_b = _prepare(image_b, region_b)

    sift = cv2.SIFT_create(MAX_KEYPOINTS)
    points_a, descriptors_a = sift.detectAndCompute(gray_a, mask_a)
    points_b, descriptors_b = sift.detectAndCompute(gray_b, mask_b)
    homography, points = _fit_homography(points_a, descriptors_a, points_b, descriptors_b)

    overlap = 0.0
    if homography is not None:
        moved = cv2.warpPerspective(mask_a, homography, gray_b.shape[::-1], flags=cv2.INTER_NEAREST) > 0
        target = mask_b > 0
        union = np.count_nonzero(moved | target)
        overlap = round(np.count_nonzero(moved & target) / union, 4) if union else 0.0

    comparable = bool(points >= min_points and overlap >= min_overlap)
    return ViewCheck(comparable, points, float(overlap), min_points, min_overlap)
