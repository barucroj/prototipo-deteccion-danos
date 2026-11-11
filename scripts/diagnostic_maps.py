"""One diagnostic image per A/B pair: the whole system (M1 -> M4) on a folder of pairs.

Each image has a header (pair, view check, what the system said and, with
labels, what was real) above two panels: A with the M3 detections and B with
the M4 change map. Labeled boxes are drawn in green. The comparison is always
forced so every pair gets a map; the header says whether the view check
would have blocked it.

    ./venv_cuda/Scripts/python.exe scripts/diagnostic_maps.py data/raw/own-data/batch_final -o salida/

Without ``annotations/instances_images.json`` the pairs are found by name
(``<name>_A.jpg`` / ``<name>_B.jpg``) and the header has no ground truth.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import textwrap
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.app.evaluation_view import draw_ground_truth  # noqa: E402
from src.evaluation.ground_truth import load_batch  # noqa: E402
from src.evaluation.metrics import evaluate_pair  # noqa: E402
from src.m3_damage.assignment.analyze import DEFAULT_DAMAGE_THRESHOLD  # noqa: E402
from src.m3_damage.assignment.render import draw_analysis  # noqa: E402
from src.pipeline.orchestrator import PairPipeline, PipelineSettings  # noqa: E402

PANEL_HEIGHT = 1000
FONT = cv2.FONT_HERSHEY_SIMPLEX
TYPES = {"dent": "golpe", "scratch": "rayon", "crack": "fisura", "glass shatter": "cristal roto",
         "lamp broken": "faro roto", "tire flat": "llanta ponchada"}
STATES = {"nuevo": "nuevo", "preexistente": "preexistente", "no_visible": "no visible",
          "no_detectado": "NO DETECTADO", "emparejado_mal": "emparejado mal", "sin_clasificar": "sin clasificar"}


def find_pairs(folder: str):
    """Pairs of ``folder``: from the COCO labels if they exist, else by file name."""
    annotations = os.path.join(folder, "annotations", "instances_images.json")
    if os.path.exists(annotations):
        return load_batch(folder), True
    images = os.path.join(folder, "images") if os.path.isdir(os.path.join(folder, "images")) else folder
    pairs = []
    for path_a in sorted(glob.glob(os.path.join(images, "*_A.*"))):
        stem = os.path.basename(path_a).rsplit("_A.", 1)[0]
        matches = glob.glob(os.path.join(images, f"{stem}_B.*"))
        if matches:
            pairs.append(SimpleNamespace(par=stem, vista="", path_a=path_a, path_b=matches[0]))
    return pairs, False


def fit_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    return cv2.resize(image, (int(round(image.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)


def caption(image: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((44, image.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (10, 31), FONT, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, image])


def header(lines, width: int) -> np.ndarray:
    """White band with the text lines, wrapped to the image width (ASCII only: OpenCV fonts)."""
    chars = max(40, width // 15)
    wrapped = []
    for line, color in lines:
        for i, part in enumerate(textwrap.wrap(line, chars) or [""]):
            wrapped.append(("   " + part if i else part, color))
    band = np.full((20 + 34 * len(wrapped), width, 3), 255, np.uint8)
    for row, (text, color) in enumerate(wrapped):
        cv2.putText(band, text, (14, 40 + 34 * row), FONT, 0.85, color, 2, cv2.LINE_AA)
    return band


def describe(truth, result, evaluation):
    """Header lines: (text, BGR color)."""
    view = result.comparison["verificacion_vista"]
    summary = result.comparison["resumen"]
    verdict = "pasa" if view["comparable"] else "NO PASA -> comparacion forzada"
    lines = [(f"{truth.par}  {truth.vista}   |   verificacion de vista: {view['puntos']} puntos "
              f"(min {view['umbral_puntos']}), solape {view['solape']:.2f} (min {view['umbral_solape']}) -> {verdict}",
              (0, 0, 0) if view["comparable"] else (0, 0, 200)),
             (f"Sistema: {summary['nuevos']} nuevo(s), {summary['preexistentes']} preexistente(s), "
              f"{summary['no_visibles']} no visible(s)", (0, 0, 0))]
    if evaluation is None:
        return lines
    reales = evaluation["danos"]
    new_real = [d for d in reales if d["estado_real"] == "nuevo"]
    found = sum(d["estado_predicho"] == "nuevo" for d in new_real)
    false_new = [f for f in evaluation["falsos"] if f["estado_predicho"] == "nuevo"]
    wrong_new = [d for d in reales if d["estado_real"] != "nuevo" and d["estado_predicho"] == "nuevo"]
    ok = found == len(new_real) and not false_new and not wrong_new
    lines.append((f"Etiqueta real: {len(new_real)} nuevo(s), "
                  f"{sum(d['estado_real'] == 'preexistente' for d in reales)} preexistente(s), "
                  f"{sum(d['estado_real'] == 'no_visible' for d in reales)} no visible(s)   |   "
                  f"nuevos encontrados {found}/{len(new_real)}, nuevos falsos {len(false_new) + len(wrong_new)}   "
                  f"->  {'PAR CORRECTO' if ok else 'PAR CON ERRORES'}", (0, 140, 0) if ok else (0, 0, 200)))
    for d in reales:
        iou = d["iou_b"] if d["iou_b"] is not None else d["iou_a"]
        hit = d["estado_predicho"] == d["estado_real"]
        lines.append((f"  real {TYPES.get(d['tipo'], d['tipo'])} {STATES[d['estado_real']]} -> sistema: "
                      f"{STATES.get(d['estado_predicho'], d['estado_predicho'])}"
                      + (f" como {TYPES.get(d['tipo_predicho'], d['tipo_predicho'])}" if d["tipo_predicho"] else "")
                      + (f" (IoU {iou:.2f})" if iou is not None else ""),
                      (0, 140, 0) if hit else (0, 0, 200)))
    for f in false_new:
        lines.append((f"  falso nuevo: {TYPES.get(f['tipo'], f['tipo'])} {f['confianza']:.2f} (no hay dano real ahi)",
                      (0, 0, 200)))
    return lines


def legend() -> tuple:
    return ("Verde = etiqueta real.  Mapa: rojo relleno = NUEVO, naranja tenue = PREEXISTENTE, "
            "azul punteado = NO VISIBLE.  Panel A: detecciones de M3 en A.", (90, 90, 90))


def render_pair(pipeline, truth, settings, has_labels: bool, iou: float) -> np.ndarray:
    result = pipeline.analyze_pair(truth.path_a, truth.path_b, settings, name_a=os.path.basename(truth.path_a),
                                   name_b=os.path.basename(truth.path_b), force=True)
    evaluation = None
    panel_a = draw_analysis(result.a.image, result.a.analysis)
    panel_b = result.change_map
    if has_labels:
        evaluation = evaluate_pair(truth, result.a.analysis.report, result.b.analysis.report, result.comparison, iou)
        panel_a = draw_ground_truth(panel_a, evaluation, "A")
        panel_b = draw_ground_truth(panel_b, evaluation, "B")
    panel_a = caption(fit_height(panel_a, PANEL_HEIGHT), "A (antes): danos detectados por M3")
    panel_b = caption(fit_height(panel_b, PANEL_HEIGHT), "B (despues): mapa de cambios de M4")
    gap = np.full((panel_a.shape[0], 12, 3), 255, np.uint8)
    body = np.hstack([panel_a, gap, panel_b])
    return np.vstack([header(describe(truth, result, evaluation) + [legend()], body.shape[1]), body])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="folder with images/ (and optionally annotations/instances_images.json)")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--damage-threshold", type=float, default=DEFAULT_DAMAGE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=0.3, help="IoU to count a labeled damage as found")
    parser.add_argument("--validate", action="store_true", help="enforce the minimum resolution (off by default)")
    args = parser.parse_args(argv)

    pairs, has_labels = find_pairs(args.folder)
    if not pairs:
        print("No pairs found.")
        return 1
    os.makedirs(args.output, exist_ok=True)
    settings = PipelineSettings(validate=args.validate, damage_threshold=args.damage_threshold)
    pipeline = PairPipeline.from_checkpoints()
    for truth in pairs:
        try:
            image = render_pair(pipeline, truth, settings, has_labels, args.iou)
        except Exception as error:  # one bad pair should not stop the batch
            print(f"{truth.par}: ERROR {error}")
            continue
        path = os.path.join(args.output, f"{truth.par}_{truth.vista}.jpg" if truth.vista else f"{truth.par}.jpg")
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{truth.par}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
