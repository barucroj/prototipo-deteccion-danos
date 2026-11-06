"""Write the evaluation of a batch as ``evaluacion.json`` and ``evaluacion.md``."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from src.evaluation.metrics import FALSE_STATES, PREDICTED_STATES, REAL_STATES
from src.m4_comparison.damages import NUEVO

JSON_FILE = "evaluacion.json"
MARKDOWN_FILE = "evaluacion.md"


def _fmt(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.3f}"


def _summary_markdown(min_iou: float, summary: dict, main: bool) -> List[str]:
    title = f"## IoU ≥ {min_iou}" + (" (principal)" if main else "")
    new, pairs, det = summary["nuevo"], summary["por_par"], summary["deteccion"]
    lines = [title, "",
             "### Daños nuevos", "",
             "| Precision | Recall | F1 | Nuevos reales | TP | FP | FN | Nuevos falsos por par |",
             "|---|---|---|---|---|---|---|---|",
             f"| {_fmt(new['precision'])} | {_fmt(new['recall'])} | {_fmt(new['f1'])} | {new['soporte']} | "
             f"{new['tp']} | {new['fp']} | {new['fn']} | {_fmt(new['falsos_por_par'])} |", "",
             f"FP = {new['fp_falsos']} detecciones sin daño real + {new['fp_reales']} daños reales "
             "preexistentes o no visibles marcados como nuevos.", "",
             "### Por par: ¿B tiene un daño nuevo?", "",
             f"- Accuracy: {_fmt(pairs['accuracy'])} (TP {pairs['tp']}, FP {pairs['fp']}, FN {pairs['fn']}, "
             f"TN {pairs['tn']}; {pairs['positivos']} pares con nuevo, {pairs['negativos']} sin nuevo)",
             f"- Pares exactos (todos los nuevos encontrados y ningún nuevo falso): {pairs['exactos']}/"
             f"{summary['pares']}"]
    if pairs["negativos"] == 0:
        lines.append("- Sin pares sin daño nuevo, la accuracy por par no mide los falsos positivos.")

    lines += ["", "### Informativo", "",
              f"- Daños reales: {det['reales']}; detectados: {det['detectados']} "
              f"(recall de detección {_fmt(det['recall'])})",
              f"- Detecciones falsas: {det['falsos']} ({_fmt(det['falsos_por_par'])} por par); "
              f"marcadas como nuevo: {summary['falsos'][NUEVO]}",
              f"- Tipo correcto entre los detectados: {summary['tipos']['coinciden']}/"
              f"{summary['tipos']['detectados']}",
              "", "Matriz (filas: estado real; columnas: estado asignado):", "",
              "| real \\ asignado | " + " | ".join(PREDICTED_STATES) + " |",
              "|---" * (len(PREDICTED_STATES) + 1) + "|"]
    for state in REAL_STATES:
        row = summary["matriz"][state]
        if sum(row.values()):
            lines.append(f"| {state} | " + " | ".join(str(row[p]) for p in PREDICTED_STATES) + " |")
    lines.append("| (falso) | " + " | ".join(str(summary["falsos"].get(p, 0)) if p in FALSE_STATES else "—"
                                            for p in PREDICTED_STATES) + " |")
    return lines + [""]


def _pairs_markdown(pairs: List[dict]) -> List[str]:
    lines = ["## Detalle por par", "",
             "| Par | Vista | Comparado | Daños reales | Asignados (real → asignado, IoU B) | Falsos |",
             "|---|---|---|---|---|---|"]
    for pair in pairs:
        if "error" in pair:
            lines.append(f"| {pair['par']} | {pair['vista']} | error | — | {pair['error']} | — |")
            continue
        assigned = "; ".join(f"{d['tipo']}: {d['estado_real']} → {d['estado_predicho']}"
                             + (f" ({d['iou_b']})" if d["iou_b"] is not None else "") for d in pair["danos"])
        falsos = ", ".join(f"{f['tipo']} {f['confianza']:.2f} → {f['estado_predicho']}" for f in pair["falsos"])
        lines.append(f"| {pair['par']} | {pair['vista']} | {'sí' if pair['comparable'] else 'no'} | "
                     f"{len(pair['danos'])} | {assigned or '—'} | {falsos or '—'} |")
    return lines + [""]


def save_evaluation(output_dir: str, config: dict, summaries: Dict[float, dict],
                    pairs: Dict[float, List[dict]], main_iou: float) -> Dict[str, str]:
    """Write the JSON (everything) and the Markdown (summary per IoU, detail at ``main_iou``)."""
    os.makedirs(output_dir, exist_ok=True)
    record = {"configuracion": config, "iou_principal": main_iou,
              "resultados": [{"min_iou": iou, "resumen": summaries[iou], "pares": pairs[iou]}
                             for iou in sorted(summaries)]}
    json_path = os.path.join(output_dir, JSON_FILE)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)

    lines = [f"# Evaluación A/B: {config['lote']}", "",
             f"- Pares: {summaries[main_iou]['pares']} (bloqueados por la verificación de vista: "
             f"{summaries[main_iou]['pares_bloqueados']}); forzados: {'sí' if config['forzar'] else 'no'}",
             f"- Umbrales: partes {config['umbral_partes']}, daños {config['umbral_danos']}, "
             f"distancia de partes {config['distancia_partes']}, distancia de daños {config['distancia_danos']}, "
             f"histéresis {config.get('histeresis') or 'no'}",
             ""]
    for iou in [main_iou] + sorted(i for i in summaries if i != main_iou):
        lines += _summary_markdown(iou, summaries[iou], iou == main_iou)
    lines += _pairs_markdown(pairs[main_iou])
    markdown_path = os.path.join(output_dir, MARKDOWN_FILE)
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return {"json": json_path, "markdown": markdown_path}
