"""Tables and counts the web interface builds from M3 reports."""

import numpy as np

from src.app.view import UNASSIGNED, comparison_rows, damage_rows, part_rows, to_display


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


def compared(estado, id_a, id_b, parte=None, revisar=False):
    return {"estado": estado, "tipo": "dent", "parte": parte, "confianza": 0.876, "id_a": id_a, "id_b": id_b,
            "distancia": 0.04 if id_a and id_b else None, "requiere_revision": revisar}


def test_comparison_rows_put_new_damages_first_and_translate():
    report = {"danos": [compared("no_visible", "A-02", None, "hood", True),
                        compared("preexistente", "A-01", "B-01", "hood"),
                        compared("nuevo", None, "B-02", None, True)]}
    rows = comparison_rows(report)
    assert [r["Estado"] for r in rows] == ["Nuevo", "Preexistente", "No visible"]
    assert rows[0] == {"Estado": "Nuevo", "Tipo": "Golpe", "Parte": UNASSIGNED, "Confianza": 0.88,
                       "ID A": "—", "ID B": "B-02", "Distancia": None, "Revisar": "Sí"}
    assert rows[1]["Distancia"] == 0.04 and rows[1]["Revisar"] == "No"


def test_to_display_downscales_large_images_and_converts_to_rgb():
    image = np.zeros((3000, 4000, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue in BGR
    shown = to_display(image)
    assert shown.shape == (1200, 1600, 3)
    assert shown[0, 0].tolist() == [0, 0, 255]
    assert to_display(np.zeros((100, 200, 3), dtype=np.uint8)).shape == (100, 200, 3)
