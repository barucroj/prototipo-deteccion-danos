"""M4 damage classification (new / preexisting / not visible), on synthetic M3 reports."""

from src.m4_comparison.damages import (NO_VISIBLE, NUEVO, POR_IMAGEN, POR_PARTE, PREEXISTENTE,
                                       compare_reports)

W, H = 200, 100
DOORS = [
    {"id": "p-01", "clase_parte": "front_door", "confianza": 0.9, "bbox": [0, 0, 100, 100], "area_px": 1},
    {"id": "p-02", "clase_parte": "front_door", "confianza": 0.9, "bbox": [100, 0, 200, 100], "area_px": 1},
]


def damage(damage_id, tipo, bbox, parte_id=None, pos=None):
    return {"id": damage_id, "tipo": tipo, "confianza": 0.8, "bbox": bbox, "area_px": 1,
            "parte": "front_door" if parte_id else None, "parte_id": parte_id,
            "solapamiento": 0.9 if parte_id else 0.0, "pos_relativa": pos}


def report(label, danos, partes=DOORS, width=W, height=H):
    return {"imagen": label, "width": width, "height": height, "partes": partes, "danos": danos}


def by_estado(comparison):
    return {(d.estado, d.id_a, d.id_b) for d in comparison.danos}


def test_new_preexisting_and_not_visible():
    a = report("A", [damage("A-01", "scratch", [10, 10, 30, 20], "p-01", [0.2, 0.15]),
                     damage("A-02", "dent", [60, 60, 80, 80], "p-01", [0.7, 0.7])])
    b = report("B", [damage("B-01", "scratch", [12, 10, 32, 20], "p-01", [0.22, 0.15]),
                     damage("B-02", "dent", [120, 40, 140, 60], "p-02", [0.3, 0.5])])
    comparison = compare_reports(a, b)
    assert by_estado(comparison) == {(PREEXISTENTE, "A-01", "B-01"), (NUEVO, None, "B-02"),
                                     (NO_VISIBLE, "A-02", None)}
    assert comparison.count(NUEVO) == 1
    pre = next(d for d in comparison.danos if d.estado == PREEXISTENTE)
    assert pre.metodo == POR_PARTE and pre.distancia == 0.02


def test_every_damage_appears_once_b_first():
    a = report("A", [damage("A-01", "dent", [10, 10, 20, 20], "p-01", [0.1, 0.1])])
    b = report("B", [damage("B-01", "crack", [10, 10, 20, 20], "p-01", [0.1, 0.1]),
                     damage("B-02", "dent", [10, 10, 20, 20], "p-01", [0.1, 0.1])])
    estados = compare_reports(a, b).danos
    assert [d.id_b for d in estados] == ["B-01", "B-02"]
    assert len(estados) == 2  # A-01 paired with B-02, not listed again


def test_different_types_at_the_same_place_do_not_match():
    a = report("A", [damage("A-01", "dent", [10, 10, 20, 20], "p-01", [0.15, 0.15])])
    b = report("B", [damage("B-01", "scratch", [10, 10, 20, 20], "p-01", [0.15, 0.15])])
    assert by_estado(compare_reports(a, b)) == {(NUEVO, None, "B-01"), (NO_VISIBLE, "A-01", None)}


def test_same_place_on_the_other_door_is_a_new_damage():
    # Same pos_relativa, but on the right door instead of the left one.
    a = report("A", [damage("A-01", "dent", [40, 40, 60, 60], "p-01", [0.5, 0.5])])
    b = report("B", [damage("B-01", "dent", [140, 40, 160, 60], "p-02", [0.5, 0.5])])
    assert by_estado(compare_reports(a, b)) == {(NUEVO, None, "B-01"), (NO_VISIBLE, "A-01", None)}


def test_distance_threshold_is_015_by_default():
    a = report("A", [damage("A-01", "dent", [0, 0, 10, 10], "p-01", [0.2, 0.2]),
                     damage("A-02", "crack", [0, 0, 10, 10], "p-01", [0.2, 0.2])])
    b = report("B", [damage("B-01", "dent", [0, 0, 10, 10], "p-01", [0.34, 0.2]),
                     damage("B-02", "crack", [0, 0, 10, 10], "p-01", [0.36, 0.2])])
    estados = {d.id_b: d.estado for d in compare_reports(a, b).danos if d.id_b}
    # Stage 2 sees identical boxes, so isolate stage 1 by moving B-02's box away.
    assert estados["B-01"] == PREEXISTENTE
    b["danos"][1]["bbox"] = [150, 80, 160, 90]
    estados = {d.id_b: d.estado for d in compare_reports(a, b).danos if d.id_b}
    assert estados["B-02"] == NUEVO


def test_unassigned_damages_match_by_image_position():
    a = report("A", [damage("A-01", "scratch", [100, 50, 120, 60])])
    b = report("B", [damage("B-01", "scratch", [104, 50, 124, 60])])
    [estado] = compare_reports(a, b).danos
    assert (estado.estado, estado.metodo) == (PREEXISTENTE, POR_IMAGEN)
    assert estado.distancia == 0.02


def test_damage_on_a_seam_assigned_to_different_parts_still_matches():
    # Door in A, fender in B: stage 1 cannot pair them, stage 2 does.
    fender = {"id": "p-01", "clase_parte": "front_fender", "confianza": 0.9,
              "bbox": [0, 0, 100, 100], "area_px": 1}
    a = report("A", [damage("A-01", "dent", [95, 40, 105, 50], "p-02", [0.0, 0.45])])
    b = report("B", [damage("B-01", "dent", [95, 40, 105, 50], "p-01", [1.0, 0.45])],
               partes=[fender, DOORS[1]])
    [estado] = compare_reports(a, b).danos
    assert (estado.estado, estado.metodo) == (PREEXISTENTE, POR_IMAGEN)


def test_damages_on_a_part_missing_in_a_fall_back_to_image_position():
    a = report("A", [damage("A-01", "dent", [140, 40, 160, 60])], partes=[DOORS[0]])
    b = report("B", [damage("B-01", "dent", [140, 40, 160, 60], "p-02", [0.5, 0.5])])
    [estado] = compare_reports(a, b).danos
    assert (estado.estado, estado.metodo) == (PREEXISTENTE, POR_IMAGEN)


def test_hungarian_inside_a_part_is_not_greedy():
    # Same configuration as the matching test: greedy would pair A-01 with B-01.
    a = report("A", [damage("A-01", "scratch", [0, 0, 1, 1], "p-01", [0.30, 0.5]),
                     damage("A-02", "scratch", [0, 0, 1, 1], "p-01", [0.45, 0.5])])
    b = report("B", [damage("B-01", "scratch", [199, 99, 200, 100], "p-01", [0.40, 0.5]),
                     damage("B-02", "scratch", [199, 99, 200, 100], "p-01", [0.20, 0.5])])
    assert by_estado(compare_reports(a, b)) == {(PREEXISTENTE, "A-01", "B-02"),
                                                (PREEXISTENTE, "A-02", "B-01")}


def test_empty_reports():
    comparison = compare_reports(report("A", []), report("B", []))
    assert comparison.danos == []
    assert len(comparison.partes.pares) == 2
