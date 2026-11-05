"""Training curves (loss, COCO AP, AR, LR) of the parts and damage detectors, all in one folder.

    ./venv_cuda/Scripts/python.exe scripts/plot_training_curves.py [-o data/processed/curvas_entrenamiento]

Reads the ``metrics.json`` saved by the trainer in each checkpoint folder:
``car_parts/v2`` (M2), ``damage/v2`` (M3) and ``damage/v1`` (only for the v1
vs v2 comparison). ``car_parts/v1`` is left out: it only saved a homemade
mAP@0.5, not comparable with COCO AP. Writes ``<modelo>_NN_<figura>.png``,
``damage_v1_vs_v2.png``, ``resumen.md`` and ``resumen.json`` in ``--output``.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detection_common.training_curves import (load_run, plot_comparison, plot_run,  # noqa: E402
                                                  write_summary)

CHECKPOINTS = os.path.join(PROJECT_ROOT, "models", "checkpoints")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-o", "--output", default=os.path.join(PROJECT_ROOT, "data", "processed",
                                                               "curvas_entrenamiento"))
    parser.add_argument("--parts", default=os.path.join(CHECKPOINTS, "car_parts", "v2"))
    parser.add_argument("--damage", default=os.path.join(CHECKPOINTS, "damage", "v2"))
    parser.add_argument("--damage-reference", default=os.path.join(CHECKPOINTS, "damage", "v1"),
                        help="damage version to compare against; empty to skip")
    args = parser.parse_args(argv)

    parts = load_run(args.parts, "car_parts_v2")
    damage = load_run(args.damage, "damage_v2")
    runs, figures = [parts, damage], {}
    figures["Partes (M2), car_parts/v2"] = plot_run(parts, args.output)
    figures["Daños (M3), damage/v2"] = plot_run(damage, args.output)

    if args.damage_reference:
        # damage/v1 does not save select_by; it was trained selecting by box AP.
        reference = load_run(args.damage_reference, "damage_v1", select_by="bbox")
        runs.append(reference)
        figures["Daños v1 contra v2"] = [plot_comparison(
            [reference, damage], args.output, "damage_v1_vs_v2",
            "Daños (M3): damage/v1 (sin augmentation) contra damage/v2 (con augmentation)")]

    summary = write_summary(runs, figures, args.output)
    total = sum(len(p) for p in figures.values())
    print(f"{total} figuras y {summary} en {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
