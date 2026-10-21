"""Run parts (M2) and damage (M3) detection on every image of a folder.

For each image ``foto.jpg`` it writes to the output folder:

    foto.json             M3 report (parts, damages, part each damage sits on)
    foto_anotada.jpg      damages and the outline of their part
    foto_partes.jpg       every part detected (>= part threshold)
    foto_danos.jpg        every damage detected (>= damage threshold)
    masks/foto_A-NN.png   one binary mask per damage

plus ``resumen.json`` with the counts of every image. Models are loaded once.
No GUI: safe to run unattended.

    ./venv_cuda/Scripts/python.exe scripts/analyze_folder.py [input_dir] [-o output_dir]

With no arguments it uses data/raw/own-data/TESTS/duster-condicion media-multiples angulos
and writes to its results/ subfolder.
"""

import argparse
import json
import os
import sys
import time

import cv2
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.detection_common.predict import load_checkpoint  # noqa: E402
from src.detection_common.visualize import draw_detections  # noqa: E402
from src.m1_preprocessing.pipeline import load_image  # noqa: E402
from src.m3_damage.assignment.analyze import (  # noqa: E402
    DEFAULT_DAMAGE_THRESHOLD, DEFAULT_MIN_OVERLAP, DEFAULT_PART_THRESHOLD, analyze_image)
from src.m3_damage.assignment.output import save_analysis  # noqa: E402

DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "data", "raw", "own-data", "TESTS",
                             "duster-condicion media-multiples angulos")
DEFAULT_PARTS_CHECKPOINT = os.path.join(PROJECT_ROOT, "models", "checkpoints", "car_parts", "v2", "best_model.pth")
DEFAULT_DAMAGE_CHECKPOINT = os.path.join(PROJECT_ROOT, "models", "checkpoints", "damage", "v2", "best_model.pth")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def draw_all(image_bgr, detections):
    """Every instance of one detector (mask, box, class, score), one color each."""
    indices = list(range(len(detections)))
    names = dict(enumerate(detections.names))
    return draw_detections(image_bgr.copy(), detections.boxes, indices, detections.scores,
                           names, masks=detections.masks)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input_dir", nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", default=None, help="default: <input_dir>/results")
    parser.add_argument("--parts-checkpoint", default=DEFAULT_PARTS_CHECKPOINT)
    parser.add_argument("--damage-checkpoint", default=DEFAULT_DAMAGE_CHECKPOINT)
    parser.add_argument("--part-threshold", type=float, default=DEFAULT_PART_THRESHOLD)
    parser.add_argument("--damage-threshold", type=float, default=DEFAULT_DAMAGE_THRESHOLD)
    parser.add_argument("--min-overlap", type=float, default=DEFAULT_MIN_OVERLAP)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output or os.path.join(args.input_dir, "results")

    images = sorted(name for name in os.listdir(args.input_dir)
                    if name.lower().endswith(IMAGE_EXTENSIONS)
                    and os.path.isfile(os.path.join(args.input_dir, name)))
    if not images:
        print(f"No images in {args.input_dir}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    parts_model, part_categories = load_checkpoint(args.parts_checkpoint, device)
    damage_model, damage_categories = load_checkpoint(args.damage_checkpoint, device)

    summary = {
        "input_dir": args.input_dir,
        "parts_checkpoint": args.parts_checkpoint,
        "damage_checkpoint": args.damage_checkpoint,
        "part_threshold": args.part_threshold,
        "damage_threshold": args.damage_threshold,
        "min_overlap": args.min_overlap,
        "imagenes": [],
    }
    for name in images:
        path = os.path.join(args.input_dir, name)
        stem = os.path.splitext(name)[0]
        image_bgr, _ = load_image(path)
        start = time.perf_counter()
        analysis = analyze_image(image_bgr, parts_model, part_categories, damage_model, damage_categories,
                                 device, file_name=name, part_threshold=args.part_threshold,
                                 damage_threshold=args.damage_threshold, min_overlap=args.min_overlap)
        elapsed = time.perf_counter() - start

        save_analysis(analysis, image_bgr, output_dir, stem)
        cv2.imwrite(os.path.join(output_dir, f"{stem}_partes.jpg"), draw_all(image_bgr, analysis.parts))
        cv2.imwrite(os.path.join(output_dir, f"{stem}_danos.jpg"), draw_all(image_bgr, analysis.damages))

        danos = analysis.report["danos"]
        height, width = image_bgr.shape[:2]
        print(f"\n{name} ({width}x{height}, {elapsed:.1f} s): "
              f"{len(analysis.report['partes'])} part(s), {len(danos)} damage(s)")
        for dano in danos:
            where = f"{dano['parte']} ({dano['parte_id']})" if dano["parte"] else "no part assigned"
            print(f"  {dano['id']}  {dano['tipo']:14} {dano['confianza']:.2f}  -> {where}  "
                  f"overlap {dano['solapamiento']:.2f}")
        summary["imagenes"].append({
            "imagen": name,
            "ancho": width,
            "alto": height,
            "segundos": round(elapsed, 2),
            "partes": [p["clase_parte"] for p in analysis.report["partes"]],
            "danos": [{"id": d["id"], "tipo": d["tipo"], "confianza": d["confianza"], "parte": d["parte"]}
                      for d in danos],
        })

    with open(os.path.join(output_dir, "resumen.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"\nResults in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
