"""M4 pair report, on synthetic M3 reports."""

import json

import cv2
import numpy as np

from src.m4_comparison.report import CHANGE_MAP_FILE, REPORT_FILE, build_report, expected_box_in_b, save_report
from src.m4_comparison.parts import match_parts

W, H = 200, 100
DOOR_A = {"id": "p-01", "clase_parte": "front_door", "confianza": 0.9, "bbox": [0, 0, 100, 100], "area_px": 1}
DOOR_B = {"id": "p-07", "clase_parte": "front_door", "confianza": 0.9, "bbox": [20, 0, 120, 100], "area_px": 1}


def damage(damage_id, tipo, bbox, parte_id=None, pos=None, confianza=0.8):
    return {"id": damage_id, "tipo": tipo, "confianza": confianza, "bbox": bbox, "area_px": 1,
            "parte": "front_door" if parte_id else None, "parte_id": parte_id,
            "solapamiento": 0.9 if parte_id else 0.0, "pos_relativa": pos}


def report(label, danos, partes, archivo=None, width=W, height=H):
    return {"imagen": label, "archivo": archivo, "width": width, "height": height,
            "partes": partes, "danos": danos}


def three_states():
    a = report("A", [damage("A-01", "scratch", [10, 10, 30, 20], "p-01", [0.2, 0.15], 0.6),
                     damage("A-02", "dent", [60, 60, 80, 80], "p-01", [0.7, 0.7])], [DOOR_A], "a.jpg")
    b = report("B", [damage("B-01", "scratch", [32, 10, 52, 20], "p-07", [0.22, 0.15], 0.9),
                     damage("B-02", "crack", [150, 40, 170, 60])], [DOOR_B], "b.jpg")
    return a, b


def by_state(result, estado):
    return next(d for d in result["danos"] if d["estado"] == estado)


def test_report_follows_the_data_model():
    result = build_report(*three_states())
    assert result["imagen_a"] == "a.jpg" and result["imagen_b"] == "b.jpg"
    assert result["resumen"] == {"nuevos": 1, "preexistentes": 1, "no_visibles": 1}
    assert result["mapa_cambios"] == CHANGE_MAP_FILE
    assert result["umbral_distancia"] == 0.15 and result["umbral_distancia_partes"] == 0.25
    assert result["modelos"] is None
    assert len(result["danos"]) == 3
    json.dumps(result)


def test_preexisting_damage_is_described_by_image_b():
    pre = by_state(build_report(*three_states()), "preexistente")
    assert (pre["id_a"], pre["id_b"]) == ("A-01", "B-01")
    assert pre["confianza"] == 0.9 and pre["bbox"] == [32, 10, 52, 20] and pre["bbox_en_b"] == [32, 10, 52, 20]
    assert pre["parte_id"] == "p-07" and pre["pos_relativa"] == [0.22, 0.15]
    assert pre["metodo"] == "parte" and pre["distancia"] == 0.02
    assert pre["requiere_revision"] is False


def test_new_damage_without_a_part_needs_review():
    new = by_state(build_report(*three_states()), "nuevo")
    assert new["id_a"] is None and new["parte"] is None and new["distancia"] is None
    assert new["requiere_revision"] is True


def test_not_visible_damage_needs_review_and_is_projected_through_its_part():
    missing = by_state(build_report(*three_states()), "no_visible")
    assert missing["id_b"] is None and missing["requiere_revision"] is True
    assert missing["bbox"] == [60, 60, 80, 80]  # where it was seen, in A
    assert missing["bbox_en_b"] == [80, 60, 100, 80]  # the door moved 20 px in B


def test_new_damage_on_a_part_does_not_need_review():
    a = report("A", [], [DOOR_A])
    b = report("B", [damage("B-01", "dent", [40, 40, 60, 60], "p-07", [0.3, 0.5])], [DOOR_B])
    assert build_report(a, b)["danos"][0]["requiere_revision"] is False


def test_expected_box_scales_with_the_image_when_the_part_has_no_pair():
    a = report("A", [], [])
    b = report("B", [], [], width=W * 2, height=H * 3)
    dano = damage("A-01", "dent", [10, 10, 20, 20])
    assert expected_box_in_b(dano, a, b, match_parts(a, b)) == [20, 30, 40, 60]


def test_expected_box_stays_inside_image_b():
    a = report("A", [], [DOOR_A])
    tall_door = dict(DOOR_B, bbox=[150, 0, 200, 100])
    b = report("B", [], [tall_door])
    dano = damage("A-01", "dent", [0, 0, 100, 100], "p-01", [0.5, 0.5])
    wide_a = dict(DOOR_A, bbox=[0, 0, 50, 100])
    assert expected_box_in_b(dano, report("A", [], [wide_a]), b, match_parts(report("A", [], [wide_a]), b, 1.5)) \
        == [150, 0, 200, 100]


def test_failed_view_check_blocks_the_comparison():
    vista = {"comparable": False, "puntos": 3, "solape": 0.1, "umbral_puntos": 40, "umbral_solape": 0.5}
    result = build_report(*three_states(), vista=vista)
    assert result["verificacion_vista"] == vista and result["comparacion_forzada"] is False
    assert result["danos"] == [] and result["mapa_cambios"] is None
    assert result["resumen"] == {"nuevos": 0, "preexistentes": 0, "no_visibles": 0}


def test_forced_comparison_sends_every_damage_to_review():
    vista = {"comparable": False, "puntos": 3, "solape": 0.1, "umbral_puntos": 40, "umbral_solape": 0.5}
    result = build_report(*three_states(), vista=vista, forzada=True)
    assert result["resumen"] == {"nuevos": 1, "preexistentes": 1, "no_visibles": 1}
    assert result["comparacion_forzada"] is True and result["mapa_cambios"] == CHANGE_MAP_FILE
    assert all(d["requiere_revision"] for d in result["danos"])


def test_passed_view_check_changes_nothing():
    vista = {"comparable": True, "puntos": 90, "solape": 0.8, "umbral_puntos": 40, "umbral_solape": 0.5}
    result = build_report(*three_states(), vista=vista)
    assert by_state(result, "preexistente")["requiere_revision"] is False


def test_empty_pair():
    result = build_report(report("A", [], []), report("B", [], []))
    assert result["danos"] == [] and result["resumen"] == {"nuevos": 0, "preexistentes": 0, "no_visibles": 0}


def test_blocked_pair_writes_only_the_json(tmp_path):
    vista = {"comparable": False, "puntos": 0, "solape": 0.0, "umbral_puntos": 40, "umbral_solape": 0.5}
    paths = save_report(build_report(*three_states(), vista=vista), None, str(tmp_path))
    assert paths["change_map"] is None
    assert not (tmp_path / CHANGE_MAP_FILE).exists()
    assert (tmp_path / REPORT_FILE).exists()


def test_save_report_writes_both_files(tmp_path):
    result = build_report(*three_states())
    image = np.zeros((H, W, 3), dtype=np.uint8)
    paths = save_report(result, image, str(tmp_path / "par"))

    assert paths["json"].endswith(REPORT_FILE)
    with open(paths["json"], encoding="utf-8") as handle:
        assert json.load(handle) == json.loads(json.dumps(result))
    assert cv2.imread(paths["change_map"]).shape == image.shape
