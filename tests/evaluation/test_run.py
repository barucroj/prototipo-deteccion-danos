"""Batch runner shared by the CLI and the web app, with a fake pipeline."""

from types import SimpleNamespace

import pytest

from src.evaluation import run
from src.evaluation.ground_truth import LabeledDamage, PairGroundTruth
from src.evaluation.run import evaluate_batch, evaluation_config, load_evaluation, view_status
from src.m1_preprocessing.validation import ImageValidationError
from src.pipeline.orchestrator import PipelineSettings


def m4(comparable, forzada=False, danos=()):
    return {"verificacion_vista": {"comparable": comparable}, "comparacion_forzada": forzada,
            "mapa_cambios": "mapa_cambios.png" if comparable or forzada else None, "danos": list(danos)}


def image(danos):
    return SimpleNamespace(analysis=SimpleNamespace(report={"danos": danos}), image=None)


class FakePipeline:
    checkpoints = {"partes": "p.pth", "danos": "d.pth"}

    def __init__(self, outputs):
        self.outputs = outputs

    def analyze_pair(self, path_a, path_b, settings, name_a=None, name_b=None, force=False):
        output = self.outputs[name_b]
        if isinstance(output, Exception):
            raise output
        return output


@pytest.fixture(autouse=True)
def no_files(monkeypatch):
    monkeypatch.setattr(run, "save_analysis", lambda *args: None)
    monkeypatch.setattr(run, "save_report", lambda *args: None)


def test_view_status():
    assert view_status(m4(True)) == "ok"
    assert view_status(m4(False, forzada=True)) == "forzada"
    assert view_status(m4(False)) == "bloqueada"


def test_batch_runs_every_pair_and_round_trips(tmp_path):
    ok = PairGroundTruth("p01", "x", "p01_x_A.jpg", "p01_x_B.jpg", [],
                         [LabeledDamage(1, "dent", (0, 0, 10, 10), None)])
    bad = PairGroundTruth("p02", "y", "p02_y_A.jpg", "p02_y_B.jpg", [],
                          [LabeledDamage(2, "dent", (0, 0, 10, 10), None)])
    detected = [{"id": "B-01", "tipo": "dent", "bbox": [0, 0, 10, 10], "confianza": 0.9}]
    pipeline = FakePipeline({
        "p01_x_B.jpg": SimpleNamespace(a=image([]), b=image(detected), change_map=None, comparison=m4(
            True, danos=[{"estado": "nuevo", "tipo": "dent", "id_a": None, "id_b": "B-01", "confianza": 0.9}])),
        "p02_y_B.jpg": ImageValidationError("resolucion", "La imagen es muy pequeña"),
    })
    settings = PipelineSettings()
    config = evaluation_config(str(tmp_path), None, settings, pipeline.checkpoints, False)
    seen = []
    record = evaluate_batch(pipeline, [ok, bad], settings, [0.3, 0.5], str(tmp_path), config,
                            on_pair=lambda i, truth, status: seen.append((i, truth.par, status)))

    assert seen[0] == (0, "p01", "ok") and seen[1][:2] == (1, "p02")
    summary = record["resumen"][0.3]
    assert summary["nuevo"]["tp"] == 1 and summary["pares_bloqueados"] == 1
    assert record["pares"][0.3][1]["error"]

    loaded = load_evaluation(str(tmp_path))
    assert loaded["iou_principal"] == 0.3 and sorted(loaded["resumen"]) == [0.3, 0.5]
    assert loaded["resumen"][0.3] == summary
    assert loaded["configuracion"]["umbral_danos"] == settings.damage_threshold
    assert load_evaluation(str(tmp_path / "nada")) is None
