"""Run M1 -> M4 on every pair of a labeled batch and measure it against the ground truth.

    ./venv_cuda/Scripts/python.exe scripts/evaluate_pairs.py data/raw/own-data/batch1 \
        -o data/processed/eval/batch1 [--force] [--no-validate] [--iou 0.3 0.5]

The batch is a CVAT COCO 1.0 export (see ``src/evaluation/ground_truth.py``).
Writes in ``--output``: ``evaluacion.json`` and ``evaluacion.md`` with the
metrics at each IoU (the first one is the main one), and ``pares/<pNN>/`` with
the M3 reports, annotated images, masks, M4 report and change map of each pair.
A pair that fails the view check counts its damages as ``sin_clasificar``
unless ``--force`` is passed. The web app runs the same evaluation.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.ground_truth import load_batch  # noqa: E402
from src.evaluation.metrics import DEFAULT_MIN_IOU  # noqa: E402
from src.evaluation.run import evaluate_batch, evaluation_config  # noqa: E402
from src.m3_damage.assignment.analyze import DEFAULT_DAMAGE_THRESHOLD, DEFAULT_PART_THRESHOLD  # noqa: E402
from src.m4_comparison.damages import DEFAULT_DAMAGE_MAX_DISTANCE  # noqa: E402
from src.m4_comparison.parts import DEFAULT_PART_MAX_DISTANCE  # noqa: E402
from src.m3_damage.assignment.merge import add_merge_arguments, merge_settings_from_args  # noqa: E402
from src.pipeline.orchestrator import (DEFAULT_DAMAGE_CHECKPOINT, DEFAULT_PARTS_CHECKPOINT,  # noqa: E402
                                       PairPipeline, PipelineSettings)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("batch", help="folder with images/ and annotations/instances_images.json")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--annotations", help="COCO JSON, default <batch>/annotations/instances_images.json")
    parser.add_argument("--iou", type=float, nargs="+", default=[DEFAULT_MIN_IOU, 0.5],
                        help="IoU thresholds to count a damage as detected; the first is the main one")
    parser.add_argument("--no-validate", action="store_true", help="do not enforce 1920x1080")
    parser.add_argument("--force", action="store_true", help="compare even if the view check fails")
    parser.add_argument("--parts-checkpoint", default=DEFAULT_PARTS_CHECKPOINT)
    parser.add_argument("--damage-checkpoint", default=DEFAULT_DAMAGE_CHECKPOINT)
    parser.add_argument("--part-threshold", type=float, default=DEFAULT_PART_THRESHOLD)
    parser.add_argument("--damage-threshold", type=float, default=DEFAULT_DAMAGE_THRESHOLD)
    parser.add_argument("--part-max-distance", type=float, default=DEFAULT_PART_MAX_DISTANCE)
    parser.add_argument("--damage-max-distance", type=float, default=DEFAULT_DAMAGE_MAX_DISTANCE)
    add_merge_arguments(parser)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    pairs = load_batch(args.batch, args.annotations)
    settings = PipelineSettings(validate=not args.no_validate, merge=merge_settings_from_args(args),
                                part_threshold=args.part_threshold,
                                damage_threshold=args.damage_threshold,
                                part_max_distance=args.part_max_distance,
                                damage_max_distance=args.damage_max_distance)
    pipeline = PairPipeline.from_checkpoints(args.parts_checkpoint, args.damage_checkpoint)
    config = evaluation_config(args.batch, args.annotations, settings, pipeline.checkpoints, args.force)

    def report_progress(index, truth, status):
        print(f"{truth.par}: {status}")

    record = evaluate_batch(pipeline, pairs, settings, args.iou, args.output, config, args.force,
                            on_pair=report_progress)
    print(f"Results in {record['archivos']['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
