"""Tables and counts the web interface builds from M3 reports."""

import numpy as np

from src.app.view import UNASSIGNED, count_by_type, damage_rows, part_rows, to_display, type_count_rows


def dano(id_, tipo, parte=None, parte_id=None):
    return {"id": id_, "tipo": tipo, "confianza": 0.876, "bbox": [0, 0, 1, 1], "area_px": 10,
            "parte": parte, "parte_id": parte_id, "solapamiento": 0.934 if parte else 0.1,
            "pos_relativa": [0.5, 0.5] if parte else None}


REPORT_A = {"partes": [{"id": "p-01", "clase_parte": "front_door", "confianza": 0.91, "area_px": 500}],
            "danos": [dano("A-01", "scratch", "front_door", "p-01")]}
REPORT_B = {"partes": [],
            "danos": [dano("B-01", "scratch"), dano("B-02", "dent"), dano("B-03", "scratch")]}


def test_damage_rows_translate_and_name_the_part_instance():
    assert damage_rows(REPORT_A) == [{"ID": "A-01", "Tipo": "Rayón", "Confianza": 0.88,
                                      "Parte": "front_door (p-01)", "Solapamiento": 0.93, "Área (px)": 10}]
    assert damage_rows(REPORT_B)[0]["Parte"] == UNASSIGNED


def test_part_rows():
    assert part_rows(REPORT_A) == [{"ID": "p-01", "Parte": "front_door", "Confianza": 0.91, "Área (px)": 500}]


def test_counts_per_type_cover_both_images_in_fixed_order():
    assert count_by_type(REPORT_B) == {"Rayón": 2, "Golpe": 1}
    assert type_count_rows(REPORT_A, REPORT_B) == [
        {"Tipo": "Golpe", "A (antes)": 0, "B (después)": 1},
        {"Tipo": "Rayón", "A (antes)": 1, "B (después)": 2},
    ]


def test_unknown_type_is_kept_after_the_known_ones():
    report = {"partes": [], "danos": [dano("B-01", "rust"), dano("B-02", "dent")]}
    assert [r["Tipo"] for r in type_count_rows({"partes": [], "danos": []}, report)] == ["Golpe", "rust"]


def test_to_display_downscales_large_images_and_converts_to_rgb():
    image = np.zeros((3000, 4000, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue in BGR
    shown = to_display(image)
    assert shown.shape == (1200, 1600, 3)
    assert shown[0, 0].tolist() == [0, 0, 255]
    assert to_display(np.zeros((100, 200, 3), dtype=np.uint8)).shape == (100, 200, 3)
