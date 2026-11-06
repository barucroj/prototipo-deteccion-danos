"""Damage-threshold sweep: filtering M3 results, picking the best threshold and the files."""

import csv
import json
from types import SimpleNamespace

import numpy as np

from src.evaluation.ground_truth import LabeledDamage, PairGroundTruth
from src.evaluation.plots import plot_sweep
from src.evaluation.sweep import (CSV_FILE, JSON_FILE, MARKDOWN_FILE, best_threshold, restrict_damages,
                                  save_sweep, sweep_pairs, sweep_rows)
from src.m3_damage.assignment.analyze import Detections, ImageAnalysis, build_report
from src.m3_damage.assignment.assign import assign_damages
from src.pipeline.orchestrator import PipelineSettings

W, H = 100, 100


def box_mask(x1, y1, x2, y2):
    mask = np.zeros((H, W), bool)
    mask[y1:y2, x1:x2] = True
    return mask


def analysis(label, damages):
    """M3 result with one door covering the image and ``damages`` = [(tipo, score, box)]."""
    parts = Detections(["front_door"], np.array([0.9]), np.array([[0, 0, W, H]], float), box_mask(0, 0, W, H)[None])
    dets = Detections([d[0] for d in damages], np.array([d[1] for d in damages], float),
                      np.array([d[2] for d in damages], float).reshape(-1, 4),
                      np.array([box_mask(*d[2]) for d in damages]).reshape(-1, H, W))
    assignments = assign_damages(dets.masks, parts.masks, parts.boxes, 0.5)
    report = build_report(parts, dets, assignments, label, W, H, file_name=f"{label}.jpg", damage_threshold=0.3)
    return ImageAnalysis(report=report, parts=parts, damages=dets, assignments=assignments)


def test_restrict_damages_keeps_the_confident_ones_and_renumbers():
    full = analysis("B", [("dent", 0.4, (0, 0, 10, 10)), ("scratch", 0.9, (50, 50, 70, 60))])
    kept = restrict_damages(full, 0.5)
    assert [d["tipo"] for d in kept.report["danos"]] == ["scratch"]
    assert kept.report["danos"][0]["id"] == "B-01" and kept.report["umbral_danos"] == 0.5
    assert kept.report["danos"][0]["parte_id"] == "p-01" and kept.damages.masks.shape == (1, H, W)
    assert restrict_damages(full, 0.3).report["danos"] == full.report["danos"]


def m4_ok():
    return {"verificacion_vista": {"comparable": True}, "comparacion_forzada": False}


class FakePipeline:
    checkpoints = None

    def __init__(self, pair):
        self.pair, self.settings = pair, None

    def analyze_pair(self, path_a, path_b, settings, name_a=None, name_b=None, force=False):
        self.settings = settings
        return self.pair


def sweep_fixture():
    truth = PairGroundTruth("p01", "x", "p01_x_A.jpg", "p01_x_B.jpg", [],
                            [LabeledDamage(1, "dent", (0, 0, 10, 10), None)])
    # B: the real dent at 0.6 and a false scratch at 0.4; A: nothing.
    pair = SimpleNamespace(a=SimpleNamespace(analysis=analysis("A", [])),
                           b=SimpleNamespace(analysis=analysis("B", [("dent", 0.6, (0, 0, 10, 10)),
                                                                     ("scratch", 0.4, (50, 50, 70, 60))])),
                           comparison=m4_ok())
    return truth, FakePipeline(pair)


def test_sweep_runs_the_models_once_at_the_lowest_threshold():
    truth, pipeline = sweep_fixture()
    results = sweep_pairs(pipeline, [truth], PipelineSettings(), [0.5, 0.3, 0.7], [0.3])
    assert pipeline.settings.damage_threshold == 0.3
    rows = {r["umbral_danos"]: r for r in sweep_rows(results)}
    assert (rows[0.3]["tp"], rows[0.3]["fp"], rows[0.3]["fn"]) == (1, 1, 0)
    assert (rows[0.5]["tp"], rows[0.5]["fp"], rows[0.5]["f1"]) == (1, 0, 1.0)
    assert (rows[0.7]["tp"], rows[0.7]["fn"], rows[0.7]["f1"]) == (0, 1, None)
    assert best_threshold(list(rows.values()), 0.3)["umbral_danos"] == 0.5


def test_best_threshold_breaks_ties_by_false_new_then_higher_threshold():
    rows = [{"umbral_danos": 0.3, "min_iou": 0.3, "f1": 0.6, "fp": 3},
            {"umbral_danos": 0.4, "min_iou": 0.3, "f1": 0.6, "fp": 1},
            {"umbral_danos": 0.5, "min_iou": 0.3, "f1": 0.6, "fp": 1},
            {"umbral_danos": 0.6, "min_iou": 0.5, "f1": 0.9, "fp": 0},
            {"umbral_danos": 0.7, "min_iou": 0.3, "f1": None, "fp": 0}]
    assert best_threshold(rows, 0.3)["umbral_danos"] == 0.5
    assert best_threshold(rows, 0.9) is None


def test_files_and_figures_are_written(tmp_path):
    truth, pipeline = sweep_fixture()
    results = sweep_pairs(pipeline, [truth], PipelineSettings(), [0.3, 0.5, 0.7], [0.3, 0.5])
    rows = sweep_rows(results)
    best = best_threshold(rows, 0.3)
    figures = plot_sweep(rows, best, 0.3, 1, str(tmp_path / "figuras"))
    config = {"lote": "batch", "pares": 1, "iou_principal": 0.3, "forzar": False, "umbral_partes": 0.7,
              "distancia_partes": 0.25, "distancia_danos": 0.15}
    paths = save_sweep(str(tmp_path), config, rows, best, results, figures)

    assert [p.split("figuras")[-1][1:] for p in figures] == ["prf_vs_umbral.png", "errores_vs_umbral.png",
                                                            "precision_recall.png", "f1_por_iou.png"]
    assert all((tmp_path / "figuras" / name).stat().st_size > 0 for name in
               ["prf_vs_umbral.png", "errores_vs_umbral.png", "precision_recall.png", "f1_por_iou.png"])
    with open(paths["csv"], encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 6 and table[0]["umbral_danos"] == "0.3" and table[0]["min_iou"] == "0.3"
    record = json.loads((tmp_path / JSON_FILE).read_text(encoding="utf-8"))
    assert record["mejor"]["umbral_danos"] == 0.5 and len(record["detalle"]) == 6
    markdown = (tmp_path / MARKDOWN_FILE).read_text(encoding="utf-8")
    assert "Mejor umbral" in markdown and "| 0.50 ◀ |" in markdown and "figuras/prf_vs_umbral.png" in markdown
    assert CSV_FILE in paths["csv"]


def test_sweep_applies_the_hysteresis_at_each_threshold():
    # The same dent: 0.45 in A and 0.8 in B. Without hysteresis it is new from 0.5 up.
    truth = PairGroundTruth("p01", "x", "p01_x_A.jpg", "p01_x_B.jpg",
                            [LabeledDamage(1, "dent", (0, 0, 10, 10), 1)],
                            [LabeledDamage(2, "dent", (0, 0, 10, 10), 1)])
    pair = SimpleNamespace(a=SimpleNamespace(analysis=analysis("A", [("dent", 0.45, (0, 0, 10, 10))])),
                           b=SimpleNamespace(analysis=analysis("B", [("dent", 0.8, (0, 0, 10, 10))])),
                           comparison=m4_ok())
    states = {}
    for low in (None, 0.4):
        pipeline = FakePipeline(pair)
        results = sweep_pairs(pipeline, [truth], PipelineSettings(hysteresis_threshold=low), [0.5, 0.7], [0.3])
        assert pipeline.settings.hysteresis_threshold is None
        states[low] = {t: results[t][0.3][0]["danos"][0]["estado_predicho"] for t in results}
    assert states[None] == {0.5: "nuevo", 0.7: "nuevo"}
    assert states[0.4] == {0.5: "preexistente", 0.7: "preexistente"}
