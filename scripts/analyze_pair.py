"""Run the whole pipeline on one A/B pair and write the report, without a GUI.

    ./venv_cuda/Scripts/python.exe scripts/analyze_pair.py A.jpg B.jpg -o salida/ [--no-validate]

Writes in ``--output``: the M3 report of each image (``A.json``, ``B.json``),
their annotated images and damage masks, the M4 report (``reporte_par.json``)
and the change map (``mapa_cambios.png``). ``--no-validate`` lets through photos
below 1920x1080.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.m1_preprocessing.validation import ImageValidationError  # noqa: E402
from src.m3_damage.assignment.output import save_analysis  # noqa: E402
from src.m4_comparison.damages import DEFAULT_DAMAGE_MAX_DISTANCE  # noqa: E402
from src.m4_comparison.parts import DEFAULT_PART_MAX_DISTANCE  # noqa: E402
from src.m4_comparison.report import save_report  # noqa: E402
from src.pipeline.orchestrator import (DEFAULT_DAMAGE_CHECKPOINT, DEFAULT_PARTS_CHECKPOINT,  # noqa: E402
                                       PairPipeline, PipelineSettings)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("image_a", help="before the rental")
    parser.add_argument("image_b", help="after the rental")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--no-validate", action="store_true", help="do not enforce 1920x1080")
    parser.add_argument("--parts-checkpoint", default=DEFAULT_PARTS_CHECKPOINT)
    parser.add_argument("--damage-checkpoint", default=DEFAULT_DAMAGE_CHECKPOINT)
    parser.add_argument("--part-max-distance", type=float, default=DEFAULT_PART_MAX_DISTANCE)
    parser.add_argument("--damage-max-distance", type=float, default=DEFAULT_DAMAGE_MAX_DISTANCE)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = PipelineSettings(validate=not args.no_validate, part_max_distance=args.part_max_distance,
                                damage_max_distance=args.damage_max_distance)
    pipeline = PairPipeline.from_checkpoints(args.parts_checkpoint, args.damage_checkpoint)
    try:
        pair = pipeline.analyze_pair(args.image_a, args.image_b, settings)
    except ImageValidationError as error:
        print(error)
        return 1

    save_analysis(pair.a.analysis, pair.a.image, args.output, "A")
    save_analysis(pair.b.analysis, pair.b.image, args.output, "B")
    paths = save_report(pair.comparison, pair.change_map, args.output)
    print(pair.comparison["resumen"])
    print(f"Results in {paths['json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
