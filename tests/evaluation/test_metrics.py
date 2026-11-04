"""End-to-end metrics on hand-made ground truth, M3 reports and M4 reports."""

import json

import pytest

from src.evaluation.ground_truth import LabeledDamage, PairGroundTruth
from src.evaluation.metrics import box_iou, evaluate_pair, failed_pair, match_boxes, summarize


def labeled(ann_id, tipo, bbox, dano_id=None):
    return LabeledDamage(ann_id, tipo, tuple(bbox), dano_id)


def detection(damage_id, tipo, bbox, confianza=0.9):
    return {"id": damage_id, "tipo": tipo, "bbox": bbox, "confianza": confianza}


def entry(estado, tipo, id_a=None, id_b=None, confianza=0.9):
    return {"estado": estado, "tipo": tipo, "id_a": id_a, "id_b": id_b, "confianza": confianza}


def comparison(danos, blocked=False):
    return {"danos": danos, "mapa_cambios": None if blocked else "mapa_cambios.png"}


def truth(danos_a, danos_b):
    return PairGroundTruth("p01", "x", "a.jpg", "b.jpg", danos_a, danos_b)


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_match_boxes_prefers_more_pairs_over_one_better_pair():
    # Greedy takes real 0 - detection 0 (IoU 0.90) and leaves real 1 without a valid detection.
    real = [[0, 0, 10, 10], [3, 0, 13, 10]]
    detected = [[0.5, 0, 10.5, 10], [-3, 0, 7, 10]]
    assert sorted((r, d) for r, d, _ in match_boxes(real, detected, 0.5)) == [(0, 1), (1, 0)]
    assert match_boxes(real, [], 0.5) == [] and match_boxes([], detected, 0.5) == []


def test_new_damage_detected_as_new_and_a_false_new_in_b():
    gt = truth([], [labeled(1, "dent", [100, 100, 200, 200])])
    report_a = {"danos": []}
    report_b = {"danos": [detection("B-01", "scratch", [110, 100, 210, 200]),
                          detection("B-02", "dent", [500, 500, 520, 520], 0.6)]}
    m4 = comparison([entry("nuevo", "scratch", id_b="B-01"), entry("nuevo", "dent", id_b="B-02", confianza=0.6)])
    result = evaluate_pair(gt, report_a, report_b, m4)

    assert result["comparable"] is True
    [dano] = result["danos"]
    assert dano["estado_real"] == "nuevo" and dano["estado_predicho"] == "nuevo"
    assert dano["tipo_predicho"] == "scratch" and dano["id_b"] == "B-01" and dano["iou_b"] > 0.8
    assert [f["id_b"] for f in result["falsos"]] == ["B-02"]
    json.dumps(result)

    summary = summarize([result])
    nuevo = summary["nuevo"]
    assert (nuevo["tp"], nuevo["fp"], nuevo["fn"]) == (1, 1, 0)
    assert (nuevo["fp_falsos"], nuevo["fp_reales"]) == (1, 0)
    assert nuevo["precision"] == 0.5 and nuevo["recall"] == 1.0 and nuevo["f1"] == pytest.approx(0.6667)
    assert summary["por_par"]["tp"] == 1 and summary["por_par"]["accuracy"] == 1.0
    assert summary["por_par"]["exactos"] == 0
    assert summary["deteccion"] == {"reales": 1, "detectados": 1, "recall": 1.0, "falsos": 1, "falsos_por_par": 1.0}
    assert summary["tipos"]["coinciden"] == 0 and summary["tipos"]["confusion"][0]["predicho"] == "scratch"


def test_new_damage_paired_with_a_false_detection_in_a_is_preexisting():
    gt = truth([], [labeled(1, "dent", [0, 0, 10, 10])])
    report_a = {"danos": [detection("A-01", "dent", [0, 0, 10, 10])]}
    report_b = {"danos": [detection("B-01", "dent", [0, 0, 10, 10])]}
    result = evaluate_pair(gt, report_a, report_b, comparison([entry("preexistente", "dent", "A-01", "B-01")]))
    assert result["danos"][0]["estado_predicho"] == "preexistente"
    assert result["falsos"] == []
    assert summarize([result])["nuevo"]["recall"] == 0.0


def test_missed_damage_and_low_iou_are_not_detected():
    gt = truth([], [labeled(1, "dent", [0, 0, 10, 10])])
    report_b = {"danos": [detection("B-01", "dent", [8, 8, 18, 18])]}
    result = evaluate_pair(gt, {"danos": []}, report_b, comparison([entry("nuevo", "dent", id_b="B-01")]))
    assert result["danos"][0]["estado_predicho"] == "no_detectado"
    assert len(result["falsos"]) == 1
    summary = summarize([result])
    assert summary["matriz"]["nuevo"]["no_detectado"] == 1 and summary["deteccion"]["recall"] == 0.0


def test_preexisting_needs_the_right_partner():
    gt = truth([labeled(1, "dent", [0, 0, 10, 10], 1)], [labeled(2, "dent", [0, 0, 10, 10], 1)])
    report_a = {"danos": [detection("A-01", "dent", [0, 0, 10, 10]), detection("A-02", "dent", [50, 50, 60, 60])]}
    report_b = {"danos": [detection("B-01", "dent", [0, 0, 10, 10])]}
    right = comparison([entry("preexistente", "dent", "A-01", "B-01"), entry("no_visible", "dent", id_a="A-02")])
    wrong = comparison([entry("preexistente", "dent", "A-02", "B-01"), entry("no_visible", "dent", id_a="A-01")])
    assert evaluate_pair(gt, report_a, report_b, right)["danos"][0]["estado_predicho"] == "preexistente"
    result = evaluate_pair(gt, report_a, report_b, wrong)
    assert result["danos"][0]["estado_predicho"] == "emparejado_mal"


def test_not_visible_damage_is_found_in_a():
    gt = truth([labeled(1, "scratch", [0, 0, 10, 10])], [])
    report_a = {"danos": [detection("A-01", "scratch", [0, 0, 10, 10])]}
    result = evaluate_pair(gt, report_a, {"danos": []}, comparison([entry("no_visible", "scratch", id_a="A-01")]))
    assert result["danos"][0]["estado_predicho"] == "no_visible" and result["danos"][0]["iou_a"] == 1.0
    summary = summarize([result])
    assert summary["matriz"]["no_visible"]["no_visible"] == 1
    assert summary["nuevo"]["soporte"] == 0 and summary["nuevo"]["precision"] is None
    assert summary["por_par"]["tn"] == 1 and summary["por_par"]["exactos"] == 1


def test_blocked_pair_leaves_damages_unclassified():
    gt = truth([], [labeled(1, "dent", [0, 0, 10, 10])])
    report_b = {"danos": [detection("B-01", "dent", [0, 0, 10, 10]), detection("B-02", "dent", [40, 40, 50, 50])]}
    result = evaluate_pair(gt, {"danos": []}, report_b, comparison([], blocked=True))
    assert result["comparable"] is False
    assert result["danos"][0]["estado_predicho"] == "sin_clasificar"
    assert [(f["id_b"], f["estado_predicho"]) for f in result["falsos"]] == [("B-02", "sin_clasificar")]
    summary = summarize([result])
    assert summary["pares_bloqueados"] == 1 and summary["nuevo"]["recall"] == 0.0
    assert summary["falsos"]["nuevo"] == 0 and summary["por_par"]["fn"] == 1


def test_failed_pair_counts_its_damages():
    gt = truth([], [labeled(1, "dent", [0, 0, 10, 10]), labeled(2, "crack", [20, 20, 30, 30])])
    result = failed_pair(gt, "resolución")
    summary = summarize([result])
    assert summary["matriz"]["nuevo"]["sin_clasificar"] == 2 and summary["pares_bloqueados"] == 1


def test_empty_batch_has_undefined_metrics():
    summary = summarize([])
    assert summary["por_par"]["accuracy"] is None
    assert summary["nuevo"] == {"precision": None, "recall": None, "f1": None, "soporte": 0, "tp": 0, "fp": 0,
                                "fn": 0, "fp_falsos": 0, "fp_reales": 0, "falsos_por_par": None}


def test_preexisting_marked_new_is_a_false_new_and_clean_pairs_are_negatives():
    # p01: a real preexisting damage that M4 calls new. p02: no new damage, a false new.
    # p03: no new damage and nothing marked new.
    pre = truth([labeled(1, "dent", [0, 0, 10, 10], 1)], [labeled(2, "dent", [0, 0, 10, 10], 1)])
    det_a = {"danos": [detection("A-01", "dent", [50, 50, 60, 60])]}
    det_b = {"danos": [detection("B-01", "dent", [0, 0, 10, 10])]}
    p01 = evaluate_pair(pre, det_a, det_b, comparison([entry("nuevo", "dent", id_b="B-01"),
                                                       entry("no_visible", "dent", id_a="A-01")]))
    p02 = evaluate_pair(truth([], []), {"danos": []}, det_b, comparison([entry("nuevo", "dent", id_b="B-01")]))
    p03 = evaluate_pair(truth([], []), {"danos": []}, {"danos": []}, comparison([]))
    summary = summarize([p01, p02, p03])
    assert (summary["nuevo"]["fp_reales"], summary["nuevo"]["fp_falsos"], summary["nuevo"]["fp"]) == (1, 1, 2)
    pairs = summary["por_par"]
    assert (pairs["positivos"], pairs["negativos"]) == (0, 3)
    assert (pairs["fp"], pairs["tn"], pairs["accuracy"], pairs["exactos"]) == (2, 1, 0.3333, 1)
