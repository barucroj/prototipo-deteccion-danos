"""Non-local means denoising of photos (``cv2.fastNlMeansDenoisingColored``).

Standalone tool to clean up a folder of photos before labeling or reviewing
them. It is **not** part of the system pipeline (``pipeline.py``, the
orchestrator and the trainer do not call it) and it does not raise the
resolution: the output has the same size as the input, with less sensor/JPEG
noise.

    ./venv_cuda/Scripts/python.exe -m src.m1_preprocessing.denoise <input_dir> <output_dir>
        [--h 5] [--h-color 5] [--template 7] [--search 21] [--quality 95]

Images are loaded upright (EXIF orientation applied, see ``load_image``) and
written with the same file name; the EXIF block is not copied.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from src.m1_preprocessing.pipeline import load_image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class DenoiseParams:
    """Arguments of ``cv2.fastNlMeansDenoisingColored``.

    ``h`` filters luminance and ``h_color`` the color channels; higher values
    remove more noise and more fine detail (thin scratches are fine detail).
    Window sizes must be odd.
    """

    h: float = 5.0
    h_color: float = 5.0
    template_window: int = 7
    search_window: int = 21

    def validate(self) -> None:
        if self.h < 0 or self.h_color < 0:
            raise ValueError("h y h_color deben ser >= 0")
        for name in ("template_window", "search_window"):
            value = getattr(self, name)
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} debe ser impar y >= 1, no {value}")


def denoise_image(image_bgr: np.ndarray, params: Optional[DenoiseParams] = None) -> np.ndarray:
    """Denoise an (H, W, 3) uint8 BGR image; returns a new image of the same size."""
    params = params or DenoiseParams()
    params.validate()
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError(f"se esperaba una imagen BGR uint8 (H, W, 3), no {image_bgr.shape} "
                         f"{image_bgr.dtype}")
    return cv2.fastNlMeansDenoisingColored(
        image_bgr, None, params.h, params.h_color, params.template_window, params.search_window)


def list_images(input_dir: str) -> List[str]:
    """JPG/PNG files directly inside ``input_dir`` (not recursive), sorted by name."""
    return sorted(os.path.join(input_dir, name) for name in os.listdir(input_dir)
                  if os.path.isfile(os.path.join(input_dir, name))
                  and name.lower().endswith(IMAGE_EXTENSIONS))


def denoise_folder(input_dir: str, output_dir: str, params: Optional[DenoiseParams] = None,
                   jpeg_quality: int = 95, verbose: bool = True) -> List[str]:
    """Denoise every image of ``input_dir`` into ``output_dir``; returns the written paths."""
    params = params or DenoiseParams()
    params.validate()
    os.makedirs(output_dir, exist_ok=True)
    written = []
    paths = list_images(input_dir)
    for i, path in enumerate(paths, 1):
        start = time.perf_counter()
        image, _ = load_image(path)
        result = denoise_image(image, params)
        out_path = os.path.join(output_dir, os.path.basename(path))
        ext = os.path.splitext(out_path)[1].lower()
        flags = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if ext in (".jpg", ".jpeg") else []
        ok, encoded = cv2.imencode(ext if ext != ".jpeg" else ".jpg", result, flags)
        if not ok:
            raise RuntimeError(f"no se pudo codificar {out_path}")
        encoded.tofile(out_path)  # also works with non-ASCII paths on Windows
        written.append(out_path)
        if verbose:
            h, w = result.shape[:2]
            print(f"[{i}/{len(paths)}] {os.path.basename(path)} {w}x{h} "
                  f"{time.perf_counter() - start:.1f} s", flush=True)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    defaults = DenoiseParams()
    parser.add_argument("--h", type=float, default=defaults.h)
    parser.add_argument("--h-color", type=float, default=defaults.h_color)
    parser.add_argument("--template", type=int, default=defaults.template_window)
    parser.add_argument("--search", type=int, default=defaults.search_window)
    parser.add_argument("--quality", type=int, default=95, help="calidad JPEG de salida")
    args = parser.parse_args(argv)
    params = DenoiseParams(args.h, args.h_color, args.template, args.search)
    written = denoise_folder(args.input_dir, args.output_dir, params, args.quality)
    print(f"{len(written)} imagenes en {args.output_dir}")
    print(f"parametros: {asdict(params)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
