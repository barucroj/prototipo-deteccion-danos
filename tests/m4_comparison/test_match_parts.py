"""M4 part matching between A and B, on synthetic M3 reports (no models)."""

import pytest

from src.m4_comparison.matching import match_points
from src.m4_comparison.parts import box_center, match_parts


def part(part_id, clase, bbox):
    return {"id": part_id, "clase_parte": clase, "confianza": 0.9, "bbox": bbox, "area_px": 1}


def report(partes, width=200, height=100):
    return {"width": width, "height": height, "partes": partes, "danos": []}


def test_match_points_prefers_the_global_optimum_over_greedy():
    # Greedy from A0 would take B0 (0.10) and leave A1 with B1 at 0.30.
    # The optimum pairs A0-B1 (0.15) and A1-B0 (0.05).
    a = [(0.0, 0.0), (0.15, 0.0)]
    b = [(0.10, 0.0), (-0.15, 0.0)]
    assert match_points(a, b, max_distance=1.0) == [(0, 1, pytest.approx(0.15)),
                                                    (1, 0, pytest.approx(0.05))]


def test_match_points_drops_pairs_beyond_the_threshold():
    assert match_points([(0.0, 0.0)], [(0.5, 0.0)], max_distance=0.25) == []


def test_match_points_maximizes_valid_pairs():
    # Distances: A0-B0 0.01, A0-B1 0.20, A1-B0 0.19, A1-B1 0.32 (> 0.21).
    # Plain minimum sum picks A0-B0 + A1-B1 (0.33) and filtering then leaves
    # one pair; gating first keeps both valid pairs A0-B1 + A1-B0.
    a = [(0.0, 0.0), (0.19, -0.06)]
    b = [(0.01, 0.0), (0.0, 0.2)]
    pairs = match_points(a, b, max_distance=0.21)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1), (1, 0)]


def test_match_points_empty_side():
    assert match_points([], [(0.1, 0.1)], 0.5) == []
    assert match_points([(0.1, 0.1)], [], 0.5) == []


def test_box_center_is_normalized_to_the_image():
    assert box_center([0, 0, 100, 50], 200, 100) == (0.25, 0.25)


def test_two_doors_of_the_same_class_keep_their_side():
    # B lists the doors in the opposite order; position, not order or name, decides.
    a = report([part("p-01", "front_door", [0, 0, 100, 100]),
                part("p-02", "front_door", [100, 0, 200, 100])])
    b = report([part("p-01", "front_door", [105, 0, 200, 100]),
                part("p-02", "front_door", [5, 0, 100, 100])])
    matching = match_parts(a, b)
    assert matching.a_to_b() == {"p-01": "p-02", "p-02": "p-01"}
    assert matching.solo_a == [] and matching.solo_b == []


def test_parts_of_different_classes_never_match():
    a = report([part("p-01", "hood", [0, 0, 100, 100])])
    b = report([part("p-01", "front_door", [0, 0, 100, 100])])
    matching = match_parts(a, b)
    assert matching.pares == []
    assert matching.solo_a == ["p-01"] and matching.solo_b == ["p-01"]


def test_part_only_in_a_is_left_unmatched():
    a = report([part("p-01", "hood", [0, 0, 100, 50]), part("p-02", "bumper", [0, 50, 200, 100])])
    b = report([part("p-01", "bumper", [0, 50, 200, 100])])
    matching = match_parts(a, b)
    assert matching.a_to_b() == {"p-02": "p-01"}
    assert matching.solo_a == ["p-01"] and matching.solo_b == []


def test_far_part_of_the_same_class_is_not_matched():
    a = report([part("p-01", "wheel", [0, 0, 40, 40])])
    b = report([part("p-01", "wheel", [160, 60, 200, 100])])
    matching = match_parts(a, b, max_distance=0.25)
    assert matching.pares == []


def test_different_resolutions_compare_in_normalized_space():
    a = report([part("p-01", "hood", [0, 0, 100, 50])], width=200, height=100)
    b = report([part("p-01", "hood", [0, 0, 1000, 500])], width=2000, height=1000)
    matching = match_parts(a, b, max_distance=0.01)
    assert matching.pares[0].distancia == 0.0
