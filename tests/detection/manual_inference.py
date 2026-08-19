"""Manual, interactive check of a trained car-parts detector checkpoint.

Opens a native file-picker window so you can choose any photo on your PC,
runs the detector on it, and shows the result with predicted boxes/labels
drawn on top.

Run directly — this opens GUI windows, so it is NOT a pytest test and is not
collected by `pytest`/`python -m pytest tests/`:

    python tests/detection/manual_inference.py

To evaluate a different checkpoint (e.g. a full training run instead of the
`models/checkpoints/test/` smoke-test one), just change CHECKPOINT_PATH below.
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

from src.detection.model import build_model  # noqa: E402

# Checkpoint to evaluate. Relative paths are resolved against the project
# root, so this can stay relative regardless of where you run the script
# from. Point this at any checkpoint saved by `python -m src.detection.train`
# (it has `best_model.pth` and `last_model.pth`).
CHECKPOINT_PATH = "models/checkpoints/test/best_model.pth"

# Minimum confidence score for a detection to be drawn/reported.
SCORE_THRESHOLD = 0.3


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(num_classes=checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    categories = {int(k): v for k, v in checkpoint["categories"].items()}
    return model, categories


def draw_predictions(image_bgr, boxes, labels, scores, categories):
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        name = categories.get(int(label), str(int(label)))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{name} {score:.2f}"
        cv2.putText(image_bgr, text, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return image_bgr


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
    checkpoint_path = CHECKPOINT_PATH
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

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(image_rgb).to(device)

    with torch.no_grad():
        output = model([tensor])[0]

    keep = output["scores"] >= SCORE_THRESHOLD
    boxes = output["boxes"][keep].cpu().numpy()
    labels = output["labels"][keep].cpu().numpy()
    scores = output["scores"][keep].cpu().numpy()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image: {image_path}")
    print(f"{len(boxes)} detection(s) >= {SCORE_THRESHOLD}:")
    for label, score in zip(labels, scores):
        print(f"  {categories.get(int(label), int(label))}: {score:.2f}")

    annotated = draw_predictions(image_bgr.copy(), boxes, labels, scores, categories)

    cv2.imshow("Detections (press any key to close)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
