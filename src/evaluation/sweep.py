"""Sweep the damage confidence threshold of M3 over a labeled batch.

The detectors run once per pair, at the lowest threshold. A higher threshold
keeps a subset of those detections, and a damage's part assignment does not
depend on the other damages, so each threshold only filters the detections and
repeats M4; duplicates are merged again after filtering. The view check
depends on the parts alone, so its verdict is reused for every threshold.
With hysteresis the detectors run down to its threshold, and at each damage
threshold the detections between both are the support of M4.

The best threshold maximizes the F1 of new damages at the main IoU; ties go to
fewer false new damages, then to the higher threshold.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence


from src.evaluation.ground_truth import PairGroundTruth
from src.evaluation.metrics import evaluate_pair, failed_pair, summarize
from src.m1_preprocessing.validation import ImageValidationError
from src.m3_damage.assignment.analyze import restrict_damages, split_support  # noqa: F401 (re-exported)
from src.m4_comparison.report import build_report as build_comparison

CSV_FILE = "barrido_umbral.csv"
JSON_FILE = "barrido_umbral.json"
MARKDOWN_FILE = "barrido_umbral.md"
DEFAULT_THRESHOLDS = tuple(round(0.30 + 0.05 * i, 2) for i in range(14))  # 0.30 .. 0.95

CSV_COLUMNS = ["umbral_danos", "min_iou", "precision", "recall", "f1", "nuevos_reales", "tp", "fp", "fn",
               "fp_falsos", "fp_reales", "nuevos_falsos_por_par", "pares_exactos", "accuracy_par",
               "detectados", "recall_deteccion", "detecciones_falsas"]


def sweep_pairs(pipeline, pairs: List[PairGroundTruth], settings, thresholds: Sequence[float],
                ious: Sequence[float], force: bool = False,
                on_pair: Optional[Callable[[int, PairGroundTruth, str], None]] = None) -> Dict[float, Dict[float, list]]:
    """Evaluate every pair at every threshold and IoU.

    Returns:
        ``{threshold: {iou: [evaluate_pair result per pair]}}``.
    """
    thresholds = sorted(thresholds)
    low = settings.hysteresis_threshold
    lowest = thresholds[0] if low is None else min(thresholds[0], low)
    # Run the detectors once, at the lowest threshold, with no hysteresis of their own.
    base = replace(settings, damage_threshold=lowest, hysteresis_threshold=None)
    results = {t: {iou: [] for iou in ious} for t in thresholds}
    for index, truth in enumerate(pairs):
        name_a, name_b = os.path.basename(truth.path_a), os.path.basename(truth.path_b)
        try:
            pair = pipeline.analyze_pair(truth.path_a, truth.path_b, base, name_a=name_a, name_b=name_b,
                                         force=force)
        except ImageValidationError as error:
            for t in thresholds:
                for iou in ious:
                    results[t][iou].append(failed_pair(truth, str(error)))
            if on_pair:
                on_pair(index, truth, str(error))
            continue

        vista = pair.comparison["verificacion_vista"]
        for t in thresholds:
            support = {}
            if low is not None and low < t:
                full_a, full_b = restrict_damages(pair.a.analysis, low), restrict_damages(pair.b.analysis, low)
                (main_a, sup_a), (main_b, sup_b) = split_support(full_a, t), split_support(full_b, t)
                report_a, report_b = main_a.report, main_b.report
                support = {"apoyo_a": sup_a.report["danos"], "apoyo_b": sup_b.report["danos"], "umbral_apoyo": low}
            else:
                report_a = restrict_damages(pair.a.analysis, t).report
                report_b = restrict_damages(pair.b.analysis, t).report
            comparison = build_comparison(report_a, report_b, settings.part_max_distance,
                                          settings.damage_max_distance, modelos=pipeline.checkpoints,
                                          vista=vista, forzada=pair.comparison["comparacion_forzada"], **support)
            for iou in ious:
                results[t][iou].append(evaluate_pair(truth, report_a, report_b, comparison, iou))
        if on_pair:
            on_pair(index, truth, "ok" if vista["comparable"] else
                    ("forzada" if pair.comparison["comparacion_forzada"] else "bloqueada"))
    return results


def sweep_rows(results: Dict[float, Dict[float, list]]) -> List[dict]:
    """One row per threshold and IoU with the metrics of new damages (the CSV)."""
    rows = []
    for t in sorted(results):
        for iou in sorted(results[t]):
            s = summarize(results[t][iou])
            new, pairs, det = s["nuevo"], s["por_par"], s["deteccion"]
            rows.append({"umbral_danos": t, "min_iou": iou, "precision": new["precision"], "recall": new["recall"],
                         "f1": new["f1"], "nuevos_reales": new["soporte"], "tp": new["tp"], "fp": new["fp"],
                         "fn": new["fn"], "fp_falsos": new["fp_falsos"], "fp_reales": new["fp_reales"],
                         "nuevos_falsos_por_par": new["falsos_por_par"], "pares_exactos": pairs["exactos"],
                         "accuracy_par": pairs["accuracy"], "detectados": det["detectados"],
                         "recall_deteccion": det["recall"], "detecciones_falsas": det["falsos"]})
    return rows


def best_threshold(rows: List[dict], iou: float) -> Optional[dict]:
    """Row with the highest F1 at ``iou``; ties: fewer false new damages, then the higher threshold."""
    candidates = [r for r in rows if r["min_iou"] == iou and r["f1"] is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["f1"], -r["fp"], r["umbral_danos"]))


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def save_sweep(output_dir: str, config: dict, rows: List[dict], best: Optional[dict],
               results: Dict[float, Dict[float, list]], figures: Sequence[str] = ()) -> Dict[str, str]:
    """Write the CSV (one row per threshold and IoU), the JSON (plus per-pair detail) and a Markdown summary."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, CSV_FILE)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(output_dir, JSON_FILE)
    detail = [{"umbral_danos": t, "min_iou": iou, "pares": results[t][iou]}
              for t in sorted(results) for iou in sorted(results[t])]
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"configuracion": config, "mejor": best, "filas": rows, "detalle": detail}, handle,
                  indent=2, ensure_ascii=False)

    main_iou = config["iou_principal"]
    lines = [f"# Barrido del umbral de confianza de daños: {config['lote']}", "",
             f"- Pares: {config['pares']}; IoU principal {main_iou}; forzados: {'sí' if config['forzar'] else 'no'}",
             f"- Fijos: umbral de partes {config['umbral_partes']}, distancia de partes {config['distancia_partes']}, "
             f"distancia de daños {config['distancia_danos']}, histéresis {config.get('histeresis') or 'no'}", ""]
    if best:
        lines += [f"**Mejor umbral (F1 de nuevos, IoU ≥ {main_iou}): {best['umbral_danos']:.2f}** — "
                  f"P {_fmt(best['precision'])}, R {_fmt(best['recall'])}, F1 {_fmt(best['f1'])}, "
                  f"{best['fp']} nuevos falsos, {best['pares_exactos']}/{config['pares']} pares exactos.", "",
                  "Elegido con este mismo lote: es un umbral de calibración, no un resultado de prueba.", ""]
    for iou in sorted({r["min_iou"] for r in rows}):
        lines += [f"## IoU ≥ {iou}" + (" (principal)" if iou == main_iou else ""), "",
                  "| Umbral | Precision | Recall | F1 | TP | FP | FN | Nuevos falsos/par | Pares exactos |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in (r for r in rows if r["min_iou"] == iou):
            mark = " ◀" if best and iou == main_iou and r["umbral_danos"] == best["umbral_danos"] else ""
            lines.append(f"| {r['umbral_danos']:.2f}{mark} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | "
                         f"{_fmt(r['f1'])} | {r['tp']} | {r['fp']} | {r['fn']} | "
                         f"{_fmt(r['nuevos_falsos_por_par'])} | {r['pares_exactos']} |")
        lines.append("")
    if figures:
        lines += ["## Figuras", ""] + [f"![{os.path.basename(f)}]({os.path.relpath(f, output_dir)})"
                                       for f in figures] + [""]
    markdown_path = os.path.join(output_dir, MARKDOWN_FILE)
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).replace("\\", "/"))
    return {"csv": csv_path, "json": json_path, "markdown": markdown_path}
