"""CLI: run a trained detector over images and save annotated copies.

Usage:
    python -m src.detection.predict <checkpoint.pth> <input_dir> <output_dir>
        [-n N] [--score-threshold T]

Draws predicted boxes + category labels + confidence scores on each image
and writes the result to ``output_dir`` for visual inspection, following
the same before/after demo pattern as ``src/preprocessing/specular_removal.py``.
"""

import argparse
import glob
import os

import cv2
import torch
from torchvision.transforms import functional as F

from src.detection.model import build_model


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(num_classes=checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    categories = {int(k): v for k, v in checkpoint["categories"].items()}
    return model, categories


def _draw_predictions(image_bgr, boxes, labels, scores, categories):
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        name = categories.get(int(label), str(int(label)))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{name} {score:.2f}"
        cv2.putText(image_bgr, text, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return image_bgr


def _demo(checkpoint_path: str, input_dir: str, output_dir: str, n: int, score_threshold: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, categories = load_checkpoint(checkpoint_path, device)

    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))[:n]

    for path in paths:
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(image_rgb).to(device)

        with torch.no_grad():
            output = model([tensor])[0]

        keep = output["scores"] >= score_threshold
        boxes = output["boxes"][keep].cpu().numpy()
        labels = output["labels"][keep].cpu().numpy()
        scores = output["scores"][keep].cpu().numpy()

        annotated = _draw_predictions(image_bgr.copy(), boxes, labels, scores, categories)

        out_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(out_path, annotated)
        print(f"Saved {out_path} ({len(boxes)} detections >= {score_threshold})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo: draw detector predictions on sample images.")
    parser.add_argument("checkpoint", help="Path to a .pth checkpoint saved by src.detection.train")
    parser.add_argument("input_dir", help="Folder containing .jpg images to run inference on")
    parser.add_argument("output_dir", help="Folder to write annotated images to")
    parser.add_argument("-n", type=int, default=10, help="Number of images to process (default: 10)")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()

    _demo(args.checkpoint, args.input_dir, args.output_dir, args.n, args.score_threshold)
