"""Evaluation files written by save_evaluation."""

import json

from src.evaluation.ground_truth import LabeledDamage, PairGroundTruth
from src.evaluation.metrics import evaluate_pair, failed_pair, summarize
from src.evaluation.report import JSON_FILE, MARKDOWN_FILE, save_evaluation

CONFIG = {"lote": "batch1", "forzar": False, "umbral_partes": 0.7, "umbral_danos": 0.5,
          "distancia_partes": 0.25, "distancia_danos": 0.15}


def pairs():
    ok = PairGroundTruth("p01", "duster-front", "a", "b", [], [LabeledDamage(1, "dent", (0, 0, 10, 10), None)])
    m4 = {"danos": [{"estado": "nuevo", "tipo": "dent", "id_a": None, "id_b": "B-01", "confianza": 0.8}],
          "mapa_cambios": "mapa_cambios.png"}
    report_b = {"danos": [{"id": "B-01", "tipo": "dent", "bbox": [0, 0, 10, 10], "confianza": 0.8}]}
    bad = PairGroundTruth("p02", "ibiza-back", "a", "b", [], [LabeledDamage(2, "crack", (0, 0, 5, 5), None)])
    return [evaluate_pair(ok, {"danos": []}, report_b, m4), failed_pair(bad, "resolución insuficiente")]


def test_files_hold_every_iou_and_the_detail(tmp_path):
    results = {0.3: pairs(), 0.5: pairs()}
    summaries = {iou: summarize(r) for iou, r in results.items()}
    paths = save_evaluation(str(tmp_path), CONFIG, summaries, results, main_iou=0.3)

    record = json.loads((tmp_path / JSON_FILE).read_text(encoding="utf-8"))
    assert record["iou_principal"] == 0.3
    assert [r["min_iou"] for r in record["resultados"]] == [0.3, 0.5]
    assert record["resultados"][0]["resumen"]["nuevo"]["recall"] == 0.5

    markdown = (tmp_path / MARKDOWN_FILE).read_text(encoding="utf-8")
    assert paths["markdown"].endswith(MARKDOWN_FILE)
    assert "IoU ≥ 0.3 (principal)" in markdown and "IoU ≥ 0.5" in markdown
    assert "| p01 | duster-front | sí | 1 | dent: nuevo → nuevo (1.0) | — |" in markdown
    assert "resolución insuficiente" in markdown
