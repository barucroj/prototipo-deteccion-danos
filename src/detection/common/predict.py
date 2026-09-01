"""Shared inference driver: load a checkpoint and annotate images.

``src/detection/car_parts/predict.py`` and ``src/detection/damage/predict.py``
are thin wrappers over this, following the same before/after demo pattern as
``src/preprocessing/specular_removal.py``.

Checkpoints written by :mod:`src.detection.common.trainer` describe
themselves (``arch``, ``num_classes``, ``categories``), so nothing here needs
to know which detector produced the file.
"""

from __future__ import annotations

import argparse
import glob
import os

import cv2
import torch
from torchvision.transforms import functional as F

from src.detection.common.model import build_model
from src.detection.common.visualize import draw_detections

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Rebuild the model a checkpoint was saved from and load its weights.

    Returns:
        ``(model, categories)`` with the model in eval mode on ``device``.

    Checkpoints written before the two-detector refactor have no ``arch``
    key; those are all Faster R-CNN car-parts models, so that is the default.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", "faster_rcnn")

    model = build_model(checkpoint["num_classes"], arch=arch, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    categories = {int(k): v for k, v in checkpoint["categories"].items()}
    return model, categories


def list_images(input_dir: str, limit: int = None):
    """Sorted image paths under ``input_dir``, capped at ``limit``."""
    paths = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(input_dir, pattern)))
    paths = sorted(paths)
    return paths[:limit] if limit else paths


def build_arg_parser(cfg) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Demo: draw {cfg.name} detector predictions on sample images.")
    parser.add_argument("checkpoint", help="Path to a .pth checkpoint saved by the train CLI")
    parser.add_argument("input_dir", help="Folder containing images to run inference on")
    parser.add_argument("output_dir", help="Folder to write annotated images to")
    parser.add_argument("-n", type=int, default=10, help="Number of images to process (default: 10)")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


@torch.no_grad()
def run_prediction(args) -> int:
    """Annotate ``args.n`` images and write them to ``args.output_dir``.

    Returns:
        The number of images written.
    """
    device = torch.device(args.device)
    model, categories = load_checkpoint(args.checkpoint, device)

    os.makedirs(args.output_dir, exist_ok=True)
    paths = list_images(args.input_dir, args.n)
    written = 0

    for path in paths:
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            continue
        tensor = F.to_tensor(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).to(device)
        output = model([tensor])[0]

        keep = output["scores"] >= args.score_threshold
        boxes = output["boxes"][keep].cpu().numpy()
        labels = output["labels"][keep].cpu().numpy()
        scores = output["scores"][keep].cpu().numpy()
        masks = output["masks"][keep].cpu().numpy() if "masks" in output else None

        annotated = draw_detections(image_bgr.copy(), boxes, labels, scores, categories, masks=masks)
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        cv2.imwrite(out_path, annotated)
        written += 1
        print(f"Saved {out_path} ({len(boxes)} detections >= {args.score_threshold})")

    return written


def main(cfg):
    """Entrypoint used by both ``predict.py`` wrappers."""
    run_prediction(build_arg_parser(cfg).parse_args())
