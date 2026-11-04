"""Tables and drawing of the batch evaluation page, from plain dicts."""

import numpy as np

from src.app.evaluation_view import (GROUND_TRUTH_COLOR, draw_ground_truth, matrix_rows, pair_damage_rows,
                                     pair_rows)
from src.evaluation.metrics import summarize


def real(id_, estado_real, estado_predicho, tipo_predicho="dent", bbox_b=(10, 10, 40, 40)):
    return {"id": id_, "tipo": "dent", "estado_real": estado_real, "estado_predicho": estado_predicho,
            "tipo_predicho": tipo_predicho, "id_a": None, "id_b": "B-01" if tipo_predicho else None,
            "iou_a": None, "iou_b": 0.8 if tipo_predicho else None,
            "bbox_real_a": None, "bbox_real_b": list(bbox_b)}


PAIRS = [
    {"par": "p01", "vista": "duster-front", "comparable": True,
     "danos": [real("p01-B1", "nuevo", "nuevo")],
     "falsos": [{"id_a": None, "id_b": "B-02", "tipo": "scratch", "confianza": 0.7, "estado_predicho": "nuevo"}]},
    {"par": "p02", "vista": "ibiza-back", "comparable": False, "error": "resolución",
     "danos": [real("p02-B1", "nuevo", "sin_clasificar", tipo_predicho=None)], "falsos": []},
]


def test_matrix_rows():
    rows = matrix_rows(summarize(PAIRS))
    assert rows[0]["Real"] == "Nuevo" and rows[0]["Nuevo"] == 1 and rows[0]["Sin clasificar"] == 1
    assert rows[-1]["Real"] == "(falso)" and rows[-1]["Nuevo"] == 1 and rows[-1]["No detectado"] is None


def test_pair_rows():
    first, second = pair_rows(PAIRS)
    assert first["Comparado"] == "Sí" and first["Nuevos reales"] == 1 and first["Nuevos acertados"] == 1
    assert first["Nuevos falsos"] == 1 and first["Exacto"] == "No"
    assert second["Comparado"] == "error" and second["Error"] == "resolución" and second["Exacto"] == "No"


def test_pair_damage_rows_list_real_then_false():
    rows = pair_damage_rows(PAIRS[0])
    assert [r["Daño"] for r in rows] == ["p01-B1", "(falso)"]
    assert rows[0]["Tipo real"] == "Golpe" and rows[0]["Acierto"] == "Sí"
    assert rows[1]["Tipo detectado"] == "Rayón" and rows[1]["Estado asignado"] == "Nuevo"
    assert pair_damage_rows(PAIRS[1])[0]["Tipo detectado"] == "—"


def test_draw_ground_truth_marks_only_the_requested_side():
    image = np.zeros((100, 100, 3), np.uint8)
    drawn = draw_ground_truth(image, PAIRS[0], "B")
    assert tuple(drawn[40, 25]) == GROUND_TRUTH_COLOR
    assert image.sum() == 0
    assert draw_ground_truth(image, PAIRS[0], "A").sum() == 0
