"""Sweep the damage confidence threshold of M3 on a labeled batch and pick the best one.

    ./venv_cuda/Scripts/python.exe scripts/sweep_damage_threshold.py data/raw/own-data/batch1 \
        -o data/processed/eval/batch1_umbral [--thresholds 0.30 0.35 ...] [--iou 0.3 0.5] [--force]

The detectors run once per pair at the lowest threshold; each threshold then
filters those detections and repeats M4 (see ``src/evaluation/sweep.py``).
Writes in ``--output``: ``barrido_umbral.csv`` (one row per threshold and IoU),
``barrido_umbral.json`` (plus per-pair detail), ``barrido_umbral.md`` and the
figures in ``figuras/``. The best threshold maximizes the F1 of new damages
at the first IoU.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.ground_truth import load_batch  # noqa: E402
from src.evaluation.metrics import DEFAULT_MIN_IOU  # noqa: E402
from src.evaluation.plots import plot_sweep  # noqa: E402
from src.evaluation.run import evaluation_config  # noqa: E402
from src.evaluation.sweep import (DEFAULT_THRESHOLDS, best_threshold, save_sweep, sweep_pairs,  # noqa: E402
                                  sweep_rows)
from src.m3_damage.assignment.analyze import DEFAULT_PART_THRESHOLD  # noqa: E402
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
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
                        help="damage confidence thresholds (default 0.30 to 0.95 every 0.05)")
    parser.add_argument("--iou", type=float, nargs="+", default=[DEFAULT_MIN_IOU, 0.5],
                        help="IoU thresholds to count a damage as detected; the first is the main one")
    parser.add_argument("--no-validate", action="store_true", help="do not enforce 1920x1080")
    parser.add_argument("--force", action="store_true", help="compare even if the view check fails")
    parser.add_argument("--parts-checkpoint", default=DEFAULT_PARTS_CHECKPOINT)
    parser.add_argument("--damage-checkpoint", default=DEFAULT_DAMAGE_CHECKPOINT)
    parser.add_argument("--part-threshold", type=float, default=DEFAULT_PART_THRESHOLD)
    parser.add_argument("--part-max-distance", type=float, default=DEFAULT_PART_MAX_DISTANCE)
    parser.add_argument("--damage-max-distance", type=float, default=DEFAULT_DAMAGE_MAX_DISTANCE)
    add_merge_arguments(parser)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    thresholds = sorted({round(t, 4) for t in args.thresholds})
    pairs = load_batch(args.batch, args.annotations)
    settings = PipelineSettings(validate=not args.no_validate, merge=merge_settings_from_args(args),
                                part_threshold=args.part_threshold,
                                damage_threshold=thresholds[0], part_max_distance=args.part_max_distance,
                                damage_max_distance=args.damage_max_distance)
    pipeline = PairPipeline.from_checkpoints(args.parts_checkpoint, args.damage_checkpoint)

    def report_progress(index, truth, status):
        print(f"{truth.par}: {status}")

    results = sweep_pairs(pipeline, pairs, settings, thresholds, args.iou, args.force, on_pair=report_progress)
    rows = sweep_rows(results)
    main_iou = args.iou[0]
    best = best_threshold(rows, main_iou)

    config = evaluation_config(args.batch, args.annotations, settings, pipeline.checkpoints, args.force)
    config.update({"umbral_danos": None, "umbrales_barridos": thresholds, "iou": args.iou,
                   "iou_principal": main_iou, "pares": len(pairs),
                   "criterio": "max F1 de nuevos en el IoU principal; empate: menos FP, luego umbral mayor"})
    figures = plot_sweep(rows, best, main_iou, len(pairs), os.path.join(args.output, "figuras"))
    paths = save_sweep(args.output, config, rows, best, results, figures)

    if best:
        print(f"Best threshold {best['umbral_danos']:.2f}: P {best['precision']} R {best['recall']} "
              f"F1 {best['f1']} FP {best['fp']}")
    print(f"Results in {paths['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
