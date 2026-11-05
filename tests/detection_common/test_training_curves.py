import json
import os

import pytest

from src.detection_common.training_curves import (load_run, plot_comparison, plot_run, summary_rows,
                                                  write_summary)


def _coco(ap):
    keys = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_1", "AR_10", "AR_100")
    return {k: ap for k in keys}


def _epoch(n, ap, best=None, full=True):
    e = {"epoch": n, "train_loss": 1.0 / n, "coco": {"bbox": _coco(ap), "segm": _coco(ap - 0.01)}}
    if full:
        e.update({"loss_components": {"loss_classifier": 0.5 / n, "loss_mask": 0.3 / n}, "lr": 0.005,
                  "select_by": "segm", "is_best": best})
    return e


def _write(tmp_path, name, epochs):
    path = tmp_path / name
    path.mkdir()
    (path / "metrics.json").write_text(json.dumps(epochs), encoding="utf-8")
    return str(path)


def test_best_epoch_is_last_flagged(tmp_path):
    folder = _write(tmp_path, "v2", [_epoch(1, 0.3, True), _epoch(2, 0.5, True), _epoch(3, 0.6, False)])
    run = load_run(folder, "v2")
    assert run.select_by == "segm"
    assert run.best_epoch == 2
    assert run.has_loss_terms and run.has_lr


def test_best_epoch_without_flags_uses_select_by(tmp_path):
    folder = _write(tmp_path, "v1", [_epoch(1, 0.3, full=False), _epoch(2, 0.5, full=False),
                                     _epoch(3, 0.4, full=False)])
    with pytest.raises(ValueError, match="select_by"):
        load_run(folder, "v1")
    run = load_run(folder, "v1", select_by="bbox")
    assert run.best_epoch == 2
    assert not run.has_loss_terms and not run.has_lr


def test_plateau_start(tmp_path):
    # segm AP = bbox AP - 0.01; max 0.64, floor 0.62 -> from epoch 3 on, every epoch stays above it
    aps = [0.40, 0.55, 0.645, 0.635, 0.65, 0.64]
    run = load_run(_write(tmp_path, "p", [_epoch(i + 1, ap, True) for i, ap in enumerate(aps)]), "p")
    assert run.plateau_start() == 3
    # still climbing at the end: plateau shorter than 3 epochs
    climbing = [0.30, 0.40, 0.50, 0.60]
    run = load_run(_write(tmp_path, "c", [_epoch(i + 1, ap, True) for i, ap in enumerate(climbing)]), "c")
    assert run.plateau_start() is None


def test_rejects_metrics_without_coco(tmp_path):
    folder = _write(tmp_path, "old", [{"epoch": 1, "train_loss": 2.0, "val_map50": 0.1}])
    with pytest.raises(ValueError, match="coco"):
        load_run(folder, "old")


def test_epochs_sorted(tmp_path):
    folder = _write(tmp_path, "v2", [_epoch(2, 0.5, True), _epoch(1, 0.3, True)])
    assert load_run(folder, "v2").x == [1, 2]


def test_plots_and_summary(tmp_path):
    full = load_run(_write(tmp_path, "v2", [_epoch(1, 0.3, True), _epoch(2, 0.5, True)]), "v2")
    old = load_run(_write(tmp_path, "v1", [_epoch(1, 0.2, full=False), _epoch(2, 0.4, full=False)]), "v1",
                   select_by="bbox")
    out = tmp_path / "figs"
    full_paths = plot_run(full, str(out))
    old_paths = plot_run(old, str(out))
    assert len(full_paths) == 7  # loss, terms, AP, loss vs AP, size, recall, LR
    assert len(old_paths) == 5  # no loss terms, no LR
    comparison = plot_comparison([old, full], str(out), "cmp", "v1 vs v2")
    for p in full_paths + old_paths + [comparison]:
        assert os.path.getsize(p) > 0

    rows = summary_rows([full, old])
    assert rows[0]["mejor_epoca"] == 2 and rows[0]["segm_AP"] == pytest.approx(0.49)
    md = write_summary([full, old], {"v2": full_paths}, str(out))
    text = open(md, encoding="utf-8").read()
    assert "| v2 | 2 | 2 | segm | — |" in text
    assert json.load(open(out / "resumen.json", encoding="utf-8"))[1]["seleccion"] == "bbox"
