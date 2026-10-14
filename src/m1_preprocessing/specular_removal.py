"""Specular highlight removal preprocessing step.

Implements the non-iterative "maximum diffuse chromaticity" method of:

    Shen, H.-L., & Cai, Q.-Y. (2009). "Simple and efficient method for
    specular highlight removal." Optical Engineering, 48(2), 087007.
    https://doi.org/10.1117/1.3086617

The method assumes the dichromatic reflection model, where each pixel's
color is a linear mixture of a colored diffuse (body) reflection and an
achromatic specular (surface) reflection under (approximately) white
illumination:

    I_c = m_d * Lambda_c + m_s / 3        for c in {R, G, B}

where ``Lambda_c`` is the diffuse chromaticity (Lambda_R + Lambda_G +
Lambda_B = 1) and ``m_s`` is the total specular magnitude (split evenly
across channels because the specular reflection is assumed achromatic).

Algorithm summary
------------------
1. Build a "pseudo specular-free" image by subtracting, at every pixel,
   the minimum channel value (a coarse estimate of the specular
   contribution) and adding back the image-wide average of that minimum
   so overall brightness is preserved::

       I_sf = I - I_min + mean(I_min)

   Because the specular term is (approximately) equal in every channel,
   this subtraction cancels it out, leaving an image whose *chromaticity*
   matches the true diffuse chromaticity even in highlighted regions.

2. Compute the maximum-channel chromaticity of both the original image
   (``rho_max``, contaminated by highlights) and the pseudo
   specular-free image (``lambda_max``, ~highlight-free). Both are
   computed as ``max(R, G, B) / (R + G + B)``.

3. Solve the dichromatic model for the total diffuse magnitude ``m_d``
   at every pixel in closed form (no iteration required)::

       m_d = I_sum * (rho_max - 1/3) / (lambda_max - 1/3)

4. Recover the specular magnitude ``m_s = I_sum - m_d`` and subtract an
   even share of it (``m_s / 3``) from every channel, clipping to a
   valid range.

5. Gate the correction by pixel brightness. The dichromatic model
   cannot distinguish a true specular highlight from a naturally
   near-achromatic *diffuse* surface (e.g. black tires, gray trim,
   asphalt in the background) since both have a chromaticity close to
   ``1/3``. Real highlights are also, by definition, bright. So the
   full correction from step 4 is only applied where pixel intensity
   is above ``brightness_threshold``; below it, the original pixel is
   kept unchanged, with a smooth (not hard-edged) transition in
   between to avoid visible seams. This keeps dark/gray non-glossy
   regions untouched while still removing bright glare.

This module operates on 8-bit BGR images (OpenCV convention), matching
how the rest of the codebase reads images with ``cv2.imread``.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ShenParams:
    """Tunable parameters for :func:`remove_specular_highlights`.

    Attributes:
        epsilon: Small constant added to denominators (``I_sum`` and
            ``lambda_max - 1/3``) to avoid division by zero on black or
            perfectly achromatic pixels. Must be > 0. Default: ``1e-6``.
        smooth_ksize: Odd kernel size (in pixels) of the median blur
            applied to the ``lambda_max`` chromaticity map before it is
            used to solve for the diffuse magnitude. ``lambda_max``
            estimated from a single noisy pixel is unstable; smoothing
            it over a small neighbourhood trades some spatial precision
            for robustness to sensor noise and fine texture. Use ``0``
            or ``1`` to disable smoothing. Default: ``3``.
        diffuse_chromaticity_clip: Lower bound enforced on
            ``lambda_max``. It must stay strictly above ``1/3`` (the
            chromaticity of a fully achromatic pixel) for the
            closed-form solution to stay numerically stable; values at
            or below ``1/3`` would imply a diffuse chromaticity as flat
            as the specular term itself, which the model cannot
            separate. Default: ``0.35``.
        brightness_threshold: Mean pixel intensity, on a ``0-255``
            scale, above which a pixel is treated as a highlight
            candidate and the full specular correction from step 4 is
            applied. Pixels at or below this are left (close to)
            unchanged, since the model cannot tell a dark/gray diffuse
            surface from a highlight. Lower it to catch dimmer glare;
            raise it if diffuse gray/black regions (tires, shadows)
            are being incorrectly darkened. Default: ``180.0``.
        brightness_softness: Width, in the same ``0-255`` intensity
            units, of the smooth transition centered on
            ``brightness_threshold`` (i.e. the ramp spans
            ``[threshold - softness/2, threshold + softness/2]``).
            ``0`` makes the gate a hard cutoff, which can create
            visible seams at the boundary; larger values blend the
            correction in more gradually. Default: ``40.0``.
    """

    epsilon: float = 1e-6
    smooth_ksize: int = 3
    diffuse_chromaticity_clip: float = 0.35
    brightness_threshold: float = 180.0
    brightness_softness: float = 40.0


def _max_chromaticity(image_float: np.ndarray, epsilon: float) -> np.ndarray:
    """Per-pixel ``max(R, G, B) / (R + G + B)`` for an HxWx3 float image."""
    channel_sum = image_float.sum(axis=2)
    channel_max = image_float.max(axis=2)
    return channel_max / (channel_sum + epsilon)


def _brightness_gate(intensity: np.ndarray, threshold: float, softness: float) -> np.ndarray:
    """Smoothstep weight in [0, 1]: ~0 below the highlight threshold, ~1 above it."""
    if softness <= 0:
        return (intensity > threshold).astype(np.float64)

    edge0 = threshold - softness / 2.0
    edge1 = threshold + softness / 2.0
    t = np.clip((intensity - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def remove_specular_highlights(image: np.ndarray, params: ShenParams = None) -> np.ndarray:
    """Remove specular highlights from a single BGR image (Shen & Cai, 2009).

    Args:
        image: HxWx3 ``uint8`` array in BGR order, as returned by
            ``cv2.imread``.
        params: :class:`ShenParams` instance controlling the
            algorithm's numerical behaviour. Uses defaults when
            omitted.

    Returns:
        HxWx3 ``uint8`` array, same shape/dtype as ``image``, holding
        the estimated diffuse-only (specular-free) reflection.

    Raises:
        ValueError: If ``image`` is not an HxWx3 array.
    """
    if params is None:
        params = ShenParams()

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 image, got shape {image.shape}")

    img = image.astype(np.float64)

    # Step 1: pseudo specular-free image (cancels the achromatic specular term).
    i_min = img.min(axis=2)
    pseudo_specular_free = img - i_min[..., None] + i_min.mean()

    # Step 2: chromaticities of the original (highlighted) and pseudo
    # specular-free (diffuse-only) images.
    rho_max = _max_chromaticity(img, params.epsilon)
    lambda_max = _max_chromaticity(pseudo_specular_free, params.epsilon)

    if params.smooth_ksize and params.smooth_ksize > 1:
        lambda_max = cv2.medianBlur(
            lambda_max.astype(np.float32), params.smooth_ksize
        ).astype(np.float64)

    lambda_max = np.clip(lambda_max, params.diffuse_chromaticity_clip, 1.0)

    # Step 3: closed-form diffuse magnitude.
    i_sum = img.sum(axis=2)
    m_d = i_sum * (rho_max - 1 / 3) / (lambda_max - 1 / 3 + params.epsilon)
    m_d = np.clip(m_d, 0, i_sum)

    # Step 4: remove an even share of the specular magnitude from each channel.
    m_s = i_sum - m_d
    corrected = img - (m_s / 3.0)[..., None]
    corrected = np.clip(corrected, 0, 255)

    # Step 5: only apply the correction to bright (highlight-candidate)
    # pixels; blend back toward the original elsewhere.
    gate = _brightness_gate(i_sum / 3.0, params.brightness_threshold, params.brightness_softness)
    diffuse = gate[..., None] * corrected + (1 - gate[..., None]) * img
    diffuse = np.clip(diffuse, 0, 255)

    return diffuse.astype(np.uint8)


def specular_map(image: np.ndarray, diffuse: np.ndarray) -> np.ndarray:
    """Grayscale visualization of the specular magnitude removed per pixel.

    Args:
        image: Original HxWx3 ``uint8`` BGR image.
        diffuse: Output of :func:`remove_specular_highlights` for
            ``image``.

    Returns:
        HxW ``uint8`` array where brighter pixels indicate a larger
        estimated specular contribution at that location. Useful for
        visually inspecting/debugging the algorithm.
    """
    original_sum = image.astype(np.int16).sum(axis=2)
    diffuse_sum = diffuse.astype(np.int16).sum(axis=2)
    removed = np.clip(original_sum - diffuse_sum, 0, 255 * 3)
    return (removed / 3).astype(np.uint8)


def _demo(input_dir: str, output_dir: str, n: int = 10) -> None:
    """Run the algorithm over up to ``n`` images and save side-by-side results."""
    import glob
    import os

    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))[:n]

    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        diffuse = remove_specular_highlights(image)
        removed = specular_map(image, diffuse)
        removed_bgr = cv2.cvtColor(removed, cv2.COLOR_GRAY2BGR)
        side_by_side = np.hstack([image, diffuse, removed_bgr])

        out_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(out_path, side_by_side)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Demo: remove specular highlights (Shen & Cai, 2009) from sample images."
    )
    parser.add_argument("input_dir", help="Folder containing .jpg images to process")
    parser.add_argument("output_dir", help="Folder to write original|diffuse|specular-map triptychs to")
    parser.add_argument("-n", type=int, default=10, help="Number of images to process (default: 10)")
    args = parser.parse_args()

    _demo(args.input_dir, args.output_dir, args.n)
