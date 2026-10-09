"""Manual, interactive check of M3 (damage + the part it is on) on any photo.

Opens a native file-picker window so you can choose a photo on your PC, runs
the parts model (M2) and the damage model on it, assigns each damage to a part
and:

1. prints the M3 JSON to the console;
2. writes the JSON, one mask PNG per damage and the annotated image to
   tests/damage_assignment/outputs/;
3. opens a window showing only the damages found and the outline of the part
   each one sits on (parts without damage are not drawn).

Defaults: car_parts/v2 with parts >= 0.7, damage/v2 with damages >= 0.5, and a
damage belongs to a part when >= 0.5 of its area lies on it. Checkpoints are
overridable as argv[1] (parts) and argv[2] (damage).

Scale caveat: CarDD is mostly close-ups of the damage. On a full-vehicle photo
thin scratches/cracks are likely missed; on a close-up the parts model may not
recognize the part, and the damage is reported as unassigned.

Run directly — this opens GUI windows, so it is NOT a pytest test:

    python tests/damage_assignment/manual_inference.py
"""

import json
import os
import sys
import tkinter as tk
from tkinter import filedialog

import cv2
import torch

# Project root = first parent folder containing src/.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(PROJECT_ROOT, "src")) and os.path.dirname(PROJECT_ROOT) != PROJECT_ROOT:
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.damage_assignment.analyze import analyze_image  # noqa: E402
from src.damage_assignment.output import save_analysis  # noqa: E402
from src.damage_assignment.render import draw_analysis  # noqa: E402
from src.detection.common.predict import load_checkpoint  # noqa: E402

# Checkpoints to use, overridable as argv[1] and argv[2]. Relative paths
# resolve against the project root, so this works from any working directory.
DEFAULT_PARTS_CHECKPOINT = os.path.join("models", "checkpoints", "car_parts", "v2", "best_model.pth")
DEFAULT_DAMAGE_CHECKPOINT = os.path.join("models", "checkpoints", "damage", "v2", "best_model.pth")

# Where the JSON, the damage masks and the annotated image are written (in .gitignore).
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tests", "damage_assignment", "outputs")

# Fraction of the screen the result window may take when it opens.
SCREEN_FRACTION = 0.85

WINDOW_NAME = "Damages and their parts (press any key to close)"


def pick_image_path():
    """Ask for an image; also return the screen size, read from the same Tk root."""
    root = tk.Tk()
    root.withdraw()
    screen = (root.winfo_screenwidth(), root.winfo_screenheight())
    path = filedialog.askopenfilename(
        title="Select a vehicle photo",
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


def resolve_checkpoint(argv_index: int, default: str) -> str:
    path = sys.argv[argv_index] if len(sys.argv) > argv_index else default
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def main():
    parts_checkpoint = resolve_checkpoint(1, DEFAULT_PARTS_CHECKPOINT)
    damage_checkpoint = resolve_checkpoint(2, DEFAULT_DAMAGE_CHECKPOINT)
    for path in (parts_checkpoint, damage_checkpoint):
        if not os.path.isfile(path):
            print(f"Checkpoint not found: {path}")
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
    parts_model, part_categories = load_checkpoint(parts_checkpoint, device)
    damage_model, damage_categories = load_checkpoint(damage_checkpoint, device)

    analysis = analyze_image(image_bgr, parts_model, part_categories, damage_model, damage_categories,
                             device, file_name=os.path.basename(image_path))

    print(json.dumps(analysis.report, indent=2, ensure_ascii=False))
    danos = analysis.report["danos"]
    print(f"\n{len(danos)} damage(s), {len(analysis.report['partes'])} part(s) detected:")
    for dano in danos:
        where = f"{dano['parte']} ({dano['parte_id']})" if dano["parte"] else "no part assigned"
        print(f"  {dano['id']}  {dano['tipo']:14} {dano['confianza']:.2f}  -> {where}  "
              f"overlap {dano['solapamiento']:.2f}")

    stem = os.path.splitext(os.path.basename(image_path))[0]
    paths = save_analysis(analysis, image_bgr, OUTPUT_DIR, stem)
    print(f"\nSaved {paths['json']}, {paths['annotated']} and {len(danos)} mask(s)")

    annotated = draw_analysis(image_bgr, analysis)
    height, width = image_bgr.shape[:2]
    win_w, win_h = fit_to_screen(width, height, screen_w, screen_h)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
    cv2.imshow(WINDOW_NAME, annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
