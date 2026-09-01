"""Manual, interactive check of a trained detector checkpoint.

Opens a native file-picker window so you can choose any photo on your PC,
runs a detector on it, and shows the result with predicted boxes, labels and
(for the damage model) masks drawn on top.

Works with either detector: checkpoints saved by the train CLIs record their
own ``arch`` and ``categories``, so the same script handles the car-parts
Faster R-CNN and the damage Mask R-CNN without being told which is which.

Run directly — this opens GUI windows, so it is NOT a pytest test and is not
collected by `pytest`/`python -m pytest tests/`:

    python tests/detection/manual_inference.py
    python tests/detection/manual_inference.py models/checkpoints/damage/v1/best_model.pth
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog

import cv2
import torch
from torchvision.transforms import functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.detection.common.predict import load_checkpoint  # noqa: E402
from src.detection.common.visualize import draw_detections  # noqa: E402

# Checkpoint to evaluate, overridable as argv[1]. Relative paths are resolved
# against the project root, so this stays valid regardless of where you run
# the script from. Point it at any checkpoint saved by a train CLI, e.g.
# models/checkpoints/damage/v1/best_model.pth for the damage detector.
DEFAULT_CHECKPOINT = os.path.join("models", "checkpoints", "car_parts", "v1", "best_model.pth")

# Minimum confidence score for a detection to be drawn/reported.
SCORE_THRESHOLD = 0.3


def pick_image_path() -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select an image to run the detector on",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
    )
    root.destroy()
    return path


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)

    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    image_path = pick_image_path()
    if not image_path:
        print("No image selected.")
        return

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Could not read image as a valid image file: {image_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, categories = load_checkpoint(checkpoint_path, device)

    tensor = F.to_tensor(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).to(device)
    with torch.no_grad():
        output = model([tensor])[0]

    keep = output["scores"] >= SCORE_THRESHOLD
    boxes = output["boxes"][keep].cpu().numpy()
    labels = output["labels"][keep].cpu().numpy()
    scores = output["scores"][keep].cpu().numpy()
    masks = output["masks"][keep].cpu().numpy() if "masks" in output else None

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image: {image_path}")
    print(f"{len(boxes)} detection(s) >= {SCORE_THRESHOLD}:")
    for label, score in zip(labels, scores):
        print(f"  {categories.get(int(label), int(label))}: {score:.2f}")

    annotated = draw_detections(image_bgr.copy(), boxes, labels, scores, categories, masks=masks)

    cv2.imshow("Detections (press any key to close)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
