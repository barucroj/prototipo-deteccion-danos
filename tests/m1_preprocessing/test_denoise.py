import cv2
import numpy as np
import pytest

from src.m1_preprocessing.denoise import DenoiseParams, denoise_folder, denoise_image, list_images


def _noisy(seed=0, size=(60, 80)):
    rng = np.random.default_rng(seed)
    clean = np.full((*size, 3), 120, np.uint8)
    noise = rng.normal(0, 15, clean.shape)
    return clean, np.clip(clean + noise, 0, 255).astype(np.uint8)


def test_denoise_reduces_noise_and_keeps_size():
    clean, noisy = _noisy()
    out = denoise_image(noisy, DenoiseParams(h=10, h_color=10))
    assert out.shape == noisy.shape and out.dtype == np.uint8
    err_before = np.abs(noisy.astype(int) - clean).mean()
    err_after = np.abs(out.astype(int) - clean).mean()
    assert err_after < err_before / 2


def test_denoise_rejects_bad_input():
    with pytest.raises(ValueError):
        denoise_image(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        denoise_image(np.zeros((10, 10, 3), np.float32))
    with pytest.raises(ValueError):
        denoise_image(np.zeros((10, 10, 3), np.uint8), DenoiseParams(template_window=6))
    with pytest.raises(ValueError):
        denoise_image(np.zeros((10, 10, 3), np.uint8), DenoiseParams(h=-1))


def test_denoise_folder(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    _, noisy = _noisy()
    cv2.imwrite(str(src / "a.JPEG"), noisy)
    cv2.imwrite(str(src / "b.png"), noisy)
    (src / "notes.txt").write_text("x")
    (src / "sub").mkdir()
    assert [p.split("\\")[-1].split("/")[-1] for p in list_images(str(src))] == ["a.JPEG", "b.png"]

    out_dir = tmp_path / "out"
    written = denoise_folder(str(src), str(out_dir), verbose=False)
    assert len(written) == 2
    for name in ("a.JPEG", "b.png"):
        image = cv2.imread(str(out_dir / name))
        assert image is not None and image.shape == noisy.shape
