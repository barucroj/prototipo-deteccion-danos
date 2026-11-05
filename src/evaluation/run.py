"""Run the A/B pipeline over a labeled batch and measure it; shared by the CLI and the web app."""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Sequence

from src.evaluation.ground_truth import PairGroundTruth
from src.evaluation.metrics import evaluate_pair, failed_pair, summarize
from src.evaluation.report import JSON_FILE, MARKDOWN_FILE, save_evaluation
from src.m1_preprocessing.validation import ImageValidationError
from src.m3_damage.assignment.output import save_analysis
from src.m4_comparison.report import save_report

PAIRS_SUBDIR = "pares"


def evaluation_config(batch_dir: str, annotations: Optional[str], settings, checkpoints: Optional[dict],
                      force: bool) -> dict:
    """What produced an evaluation, recorded in ``evaluacion.json``."""
    return {"lote": os.path.normpath(batch_dir), "anotaciones": annotations, "forzar": force,
            "validar": settings.validate, "modelos": checkpoints,
            "umbral_partes": settings.part_threshold, "umbral_danos": settings.damage_threshold,
            "solapamiento_minimo": settings.min_overlap, "fusion": settings.merge.to_json(),
            "distancia_partes": settings.part_max_distance, "distancia_danos": settings.damage_max_distance,
            "vista_puntos": settings.min_view_points, "vista_solape": settings.min_view_overlap}


def load_evaluation(output_dir: str) -> Optional[dict]:
    """Read ``evaluacion.json`` back in the shape :func:`evaluate_batch` returns; ``None`` if missing.

    The summaries are recomputed from the per-pair detail, so a file written
    by an older version of the metrics opens with the current ones.
    """
    path = os.path.join(output_dir, JSON_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    return {"configuracion": record["configuracion"], "iou_principal": record["iou_principal"],
            "resumen": {r["min_iou"]: summarize(r["pares"]) for r in record["resultados"]},
            "pares": {r["min_iou"]: r["pares"] for r in record["resultados"]},
            "archivos": {"json": path, "markdown": os.path.join(output_dir, MARKDOWN_FILE)}}


def view_status(comparison: dict) -> str:
    """``ok``, ``forzada`` or ``bloqueada`` for an M4 report."""
    if comparison["verificacion_vista"]["comparable"]:
        return "ok"
    return "forzada" if comparison["comparacion_forzada"] else "bloqueada"


def evaluate_batch(pipeline, pairs: List[PairGroundTruth], settings, ious: Sequence[float], output_dir: str,
                   config: dict, force: bool = False,
                   on_pair: Optional[Callable[[int, PairGroundTruth, str], None]] = None) -> dict:
    """Analyze every pair, write its outputs and the evaluation, and return the record.

    Args:
        pipeline: A :class:`src.pipeline.orchestrator.PairPipeline`.
        ious: IoU thresholds to count a damage as detected; the first is the main one.
        output_dir: Gets ``evaluacion.json``/``.md`` and ``pares/<pNN>/`` per pair.
        config: Settings to record in the evaluation (thresholds, batch, models).
        on_pair: Called after each pair with its index, ground truth and view
            status (``ok``, ``forzada``, ``bloqueada`` or the M1 error).

    Returns:
        ``{"configuracion", "iou_principal", "resumen": {iou: ...}, "pares": {iou: [...]}, "archivos"}``.
    """
    results: Dict[float, List[dict]] = {iou: [] for iou in ious}
    for index, truth in enumerate(pairs):
        name_a, name_b = os.path.basename(truth.path_a), os.path.basename(truth.path_b)
        try:
            pair = pipeline.analyze_pair(truth.path_a, truth.path_b, settings, name_a=name_a, name_b=name_b,
                                         force=force)
        except ImageValidationError as error:
            for iou in ious:
                results[iou].append(failed_pair(truth, str(error)))
            if on_pair:
                on_pair(index, truth, str(error))
            continue

        pair_dir = os.path.join(output_dir, PAIRS_SUBDIR, truth.par)
        save_analysis(pair.a.analysis, pair.a.image, pair_dir, "A")
        save_analysis(pair.b.analysis, pair.b.image, pair_dir, "B")
        save_report(pair.comparison, pair.change_map, pair_dir)
        for iou in ious:
            results[iou].append(evaluate_pair(truth, pair.a.analysis.report, pair.b.analysis.report,
                                              pair.comparison, iou))
        if on_pair:
            on_pair(index, truth, view_status(pair.comparison))

    summaries = {iou: summarize(results[iou]) for iou in ious}
    files = save_evaluation(output_dir, config, summaries, results, main_iou=ious[0])
    return {"configuracion": config, "iou_principal": ious[0], "resumen": summaries, "pares": results,
            "archivos": files}
