"""Manual, interactive check of the damage detector (M3) on any photo.

Opens a native file-picker window so you can choose a photo on your PC, runs
the damage Mask R-CNN on it and shows the detected damages (dent, scratch,
crack, glass shatter, lamp broken, tire flat) with masks, boxes, class and
confidence. The console lists every damage with its score and mask area.

Defaults to damage/v2 (CarDD, 6 classes, best checkpoint by mask AP). Any
checkpoint saved by a train CLI works, passed as argv[1] (e.g. damage/v1).

Scale caveat: CarDD is mostly close-ups of the damage. On a full-vehicle photo
taken with the capture protocol (4000x3000 at 2.5-4 m), the model shrinks the
image to a short side of 800 px and thin scratches/cracks end up below one
pixel wide, so they are likely missed. Close-up photos are the in-domain case.

The result window is resizable and opens scaled to fit the screen; the drawing
itself is at the original resolution.

Run directly — this opens GUI windows, so it is NOT a pytest test:

    python tests/detection/damage-segmentation/manual_inference.py
    python tests/detection/damage-segmentation/manual_inference.py models/checkpoints/damage/v1/best_model.pth
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog

import cv2
import torch
from torchvision.transforms import functional as F

# Project root = first parent folder containing src/.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(PROJECT_ROOT, "src")) and os.path.dirname(PROJECT_ROOT) != PROJECT_ROOT:
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.detection.common.predict import load_checkpoint  # noqa: E402
from src.detection.common.visualize import draw_detections  # noqa: E402

# Checkpoint to use, overridable as argv[1]. Relative paths resolve against the
# project root, so this works from any working directory.
DEFAULT_CHECKPOINT = os.path.join("models", "checkpoints", "damage", "v2", "best_model.pth")

# Minimum confidence for a damage to be drawn/reported. Same as
# tests/detection/damage-segmentation/evaluate_damage_test_set.ipynb.
SCORE_THRESHOLD = 0.5

# Fraction of the screen the result window may take when it opens.
SCREEN_FRACTION = 0.85

WINDOW_NAME = "Damages (press any key to close)"


def pick_image_path():
    """Ask for an image; also return the screen size, read from the same Tk root."""
    root = tk.Tk()
    root.withdraw()
    screen = (root.winfo_screenwidth(), root.winfo_screenheight())
    path = filedialog.askopenfilename(
        title="Select a photo to look for vehicle damage",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
    )
    root.destroy()
    return path, screen


def fit_to_screen(width: int, height: int, screen_w: int, screen_h: int,
                  fraction: float = SCREEN_FRACTION):
    """Window size that keeps the aspect ratio and fits ``fraction`` of the screen.

    Never enlarges: an image smaller than that is shown at its own size.
    """
    scale = min(1.0, fraction * screen_w / width, fraction * screen_h / height)
    return max(1, int(width * scale)), max(1, int(height * scale))


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    image_path, (screen_w, screen_h) = pick_image_path()
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

    height, width = image_bgr.shape[:2]
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image: {image_path}  ({width}x{height})")
    print(f"{len(boxes)} damage(s) >= {SCORE_THRESHOLD}:")
    for i, (label, score, box) in enumerate(zip(labels, scores, boxes)):
        area = f"{int((masks[i, 0] >= 0.5).sum())} px" if masks is not None else "-"
        x1, y1, x2, y2 = (int(v) for v in box)
        print(f"  {categories.get(int(label), int(label)):14} {score:.2f}  "
              f"box [{x1}, {y1}, {x2}, {y2}]  mask area {area}")

    annotated = draw_detections(image_bgr.copy(), boxes, labels, scores, categories, masks=masks)

    win_w, win_h = fit_to_screen(width, height, screen_w, screen_h)
    print(f"Window {win_w}x{win_h} (resizable)")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
    cv2.imshow(WINDOW_NAME, annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
