"""M4 change map, on synthetic reports and a blank image."""

import numpy as np

from src.m4_comparison.render import STATE_COLORS, draw_change_map

W, H = 400, 300


def compared(estado, box, id_a=None, id_b=None):
    return {"estado": estado, "tipo": "dent", "confianza": 0.8, "id_a": id_a, "id_b": id_b, "bbox_en_b": box}


def report(*danos):
    return {"danos": list(danos), "resumen": {"nuevos": 0, "preexistentes": 0, "no_visibles": 0}}


def blank():
    return np.full((H, W, 3), 128, dtype=np.uint8)


def mask_of(box):
    x1, y1, x2, y2 = box
    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_input_is_not_modified_and_size_is_kept():
    image = blank()
    canvas = draw_change_map(image, report())
    assert canvas.shape == image.shape
    assert np.array_equal(image, blank())


def test_new_damage_is_filled_stronger_than_a_preexisting_one():
    box = (100, 150, 200, 250)
    center = (200, 150)  # row, column inside the box
    new = draw_change_map(blank(), report(compared("nuevo", box, id_b="B-01")), {"B-01": mask_of(box)})
    old = draw_change_map(blank(), report(compared("preexistente", box, "A-01", "B-01")), {"B-01": mask_of(box)})

    gray = np.array([128, 128, 128])
    distance_new = np.abs(new[center].astype(int) - gray).sum()
    distance_old = np.abs(old[center].astype(int) - gray).sum()
    assert distance_new > distance_old > 0


def test_not_visible_damage_has_only_a_dashed_box():
    box = (100, 150, 200, 250)
    canvas = draw_change_map(blank(), report(compared("no_visible", box, id_a="A-01")))
    assert np.array_equal(canvas[200, 150], [128, 128, 128])  # no fill
    bottom_edge = canvas[250, 100:200]
    drawn = (bottom_edge == STATE_COLORS["no_visible"]).all(axis=1)
    assert 0 < drawn.sum() < len(drawn)  # dashed: part of the edge only


def test_damage_without_a_mask_still_gets_its_box():
    box = (100, 150, 200, 250)
    canvas = draw_change_map(blank(), report(compared("nuevo", box, id_b="B-01")))
    assert np.array_equal(canvas[200, 150], [128, 128, 128])
    assert (canvas[250, 150] == STATE_COLORS["nuevo"]).all()


def test_forced_comparison_carries_a_warning():
    forced = dict(report(), comparacion_forzada=True)
    assert not np.array_equal(draw_change_map(blank(), forced), draw_change_map(blank(), report()))


def test_summary_is_drawn_without_damages():
    assert not np.array_equal(draw_change_map(blank(), report()), blank())
