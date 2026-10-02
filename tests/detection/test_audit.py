"""Checks on run auditing: the live status file, ETAs, the log and the watcher.

Time is driven by a fake clock so progress and ETA numbers are exact.
"""

import json
import os

import pytest

from src.detection.common.audit import (
    LOG_FILE, RUN_INFO_FILE, STATUS_FILE, RunAuditor, dataset_summary, format_duration,
)
from src.detection.common.watch import render


class FakeClock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def read_status(tmp_path):
    with open(tmp_path / STATUS_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def run(tmp_path):
    clock = FakeClock()
    auditor = RunAuditor(str(tmp_path), total_epochs=4, update_every_s=0.0, clock=clock)
    # 30 train batches + 10 val batches -> training is 75% of an eval epoch.
    auditor.configure("car_parts", "segm", n_train_batches=30, n_val_batches=10)
    return auditor, clock


def test_format_duration():
    assert format_duration(0) == "0:00:00"
    assert format_duration(3725) == "1:02:05"
    assert format_duration(None) == "--:--:--"
    assert format_duration(float("nan")) == "--:--:--"


def test_status_tracks_epoch_batch_and_progress(run, tmp_path):
    auditor, clock = run
    auditor.start_epoch(2, will_eval=True)
    auditor.start_phase("train", 30)
    for i in range(15):
        clock.advance(1.0)
        auditor.on_batch(i, 0.5, {"loss_mask": 0.2, "loss_classifier": 0.3})

    status = read_status(tmp_path)
    assert status["state"] == "running"
    assert status["phase"] == "train"
    assert (status["epoch"], status["total_epochs"]) == (2, 4)
    assert (status["batch"], status["total_batches"]) == (15, 30)
    # Half of the training share (0.75) of the epoch.
    assert status["epoch_progress"] == pytest.approx(0.375)
    assert status["overall_progress"] == pytest.approx((1 + 0.375) / 4, abs=1e-4)
    assert status["loss_epoch_mean"] == pytest.approx(0.5)
    assert status["loss_components_epoch_mean"] == pytest.approx(
        {"loss_mask": 0.2, "loss_classifier": 0.3})


def test_eval_phase_continues_after_the_training_share(run):
    auditor, _ = run
    auditor.start_epoch(1, will_eval=True)
    auditor.start_phase("eval", 10)
    for i in range(5):
        auditor.on_batch(i)
    assert auditor.epoch_fraction() == pytest.approx(0.75 + 0.5 * 0.25)


def test_epoch_without_eval_is_all_training(run):
    auditor, _ = run
    auditor.start_epoch(1, will_eval=False)
    auditor.start_phase("train", 30)
    for i in range(30):
        auditor.on_batch(i, 1.0)
    assert auditor.epoch_fraction() == pytest.approx(1.0)


def test_eta_uses_batch_rate_then_measured_epoch_times(run):
    auditor, clock = run
    auditor.start_epoch(1, will_eval=True)
    auditor.start_phase("train", 30)
    for i in range(10):
        clock.advance(2.0)
        auditor.on_batch(i, 1.0)
    # 20 train + 10 val batches left at 2 s/batch.
    assert auditor.eta_epoch_s() == pytest.approx(60.0)

    auditor.end_epoch({"epoch": 1}, is_best=True, selected_value=0.3)
    auditor.start_epoch(2, will_eval=True)
    auditor.start_phase("train", 30)
    # Before any batch of epoch 2, the last measured epoch is the estimate.
    assert auditor.eta_epoch_s() == pytest.approx(20.0)
    # Epoch 2 + 2 more epochs at the measured 20 s per epoch.
    assert auditor.eta_total_s() == pytest.approx(20.0 + 2 * 20.0)


def test_best_epoch_is_recorded(run, tmp_path):
    auditor, _ = run
    auditor.start_epoch(1, will_eval=True)
    auditor.end_epoch({"epoch": 1}, is_best=True, selected_value=0.31)
    auditor.start_epoch(2, will_eval=True)
    auditor.end_epoch({"epoch": 2}, is_best=False, selected_value=0.29)
    assert read_status(tmp_path)["best"] == {"epoch": 1, "select_by": "segm", "value": 0.31}


def test_non_finite_losses_are_counted_not_averaged(run, tmp_path):
    auditor, _ = run
    auditor.start_epoch(1, will_eval=False)
    auditor.start_phase("train", 30)
    auditor.on_batch(0, 1.0)
    auditor.on_batch(1, float("nan"))
    assert auditor.non_finite_losses == 1
    assert auditor.loss_sum / auditor.loss_count == pytest.approx(1.0)
    assert "non-finite loss" in (tmp_path / LOG_FILE).read_text(encoding="utf-8")


def test_throttling_skips_writes_between_updates(tmp_path):
    clock = FakeClock()
    auditor = RunAuditor(str(tmp_path), total_epochs=1, update_every_s=10.0, clock=clock)
    auditor.start_epoch(1, will_eval=False)
    auditor.start_phase("train", 100)  # forced write
    auditor.on_batch(0, 1.0)           # within 10 s -> skipped
    assert read_status(tmp_path)["batch"] == 0
    clock.advance(11)
    auditor.on_batch(1, 1.0)
    assert read_status(tmp_path)["batch"] == 2


def test_finish_states_and_error_are_persisted(run, tmp_path):
    auditor, _ = run
    auditor.finish("failed", error="Traceback...\nRuntimeError: CUDA out of memory")
    status = read_status(tmp_path)
    assert status["state"] == "failed"
    assert "CUDA out of memory" in status["error"]
    assert not os.path.exists(tmp_path / (STATUS_FILE + ".tmp"))


def test_log_and_run_info_are_written(run, tmp_path):
    auditor, _ = run
    auditor.log("hello\nworld")
    auditor.write_run_info({"detector": "car_parts"})
    lines = (tmp_path / LOG_FILE).read_text(encoding="utf-8").splitlines()
    assert [line.split("] ", 1)[1] for line in lines] == ["hello", "world"]
    info = json.loads((tmp_path / RUN_INFO_FILE).read_text(encoding="utf-8"))
    assert info["detector"] == "car_parts" and "started_at" in info


class FakeDataset:
    def __init__(self, categories, samples):
        self.categories = categories
        self.samples = samples

    def __len__(self):
        return len(self.samples)


def test_dataset_summary_counts_instances_per_class():
    def anns(*ids):
        return [{"category_id": i} for i in ids]

    summary = dataset_summary(FakeDataset(
        categories={1: "door", 2: "hood", 3: "step"},
        samples=[(1, "a.jpg", 10, 10, anns(1, 1, 2)), (2, "b.jpg", 10, 10, anns(1))],
    ))
    assert summary["images"] == 2
    assert summary["instances"] == 4
    assert summary["instances_per_class"] == {"door": 3, "hood": 1}
    assert summary["classes_without_instances"] == ["step"]


def test_render_shows_epoch_progress_best_and_history(run, tmp_path):
    auditor, clock = run
    auditor.start_epoch(1, will_eval=True)
    auditor.end_epoch({"epoch": 1}, is_best=True, selected_value=0.4)
    auditor.start_epoch(2, will_eval=True)
    auditor.start_phase("train", 30)
    clock.advance(1.0)
    auditor.on_batch(0, 0.7, {"loss_mask": 0.3})

    history = [{"epoch": 1, "train_loss": 0.9, "epoch_time_s": 20.0,
                "coco": {"bbox": {"AP": 0.45}, "segm": {"AP": 0.4}}}]
    text = render(read_status(tmp_path), history, now=clock())
    assert "Epoch 2/4" in text
    assert "batch 1/30" in text
    assert "Best so far: epoch 1" in text
    assert "0.4000" in text and "0.4500" in text
    assert "WARNING: no update" not in text


def test_render_flags_a_stale_run(run, tmp_path):
    auditor, clock = run
    auditor.start_epoch(1, will_eval=True)
    auditor.start_phase("train", 30)
    text = render(read_status(tmp_path), now=clock() + 600)
    assert "WARNING: no update" in text


def test_completed_run_reports_full_progress(run, tmp_path):
    auditor, _ = run
    for epoch in range(1, 5):
        auditor.start_epoch(epoch, will_eval=True)
        auditor.end_epoch({"epoch": epoch}, is_best=False, selected_value=0.1)
    auditor.finish("completed")
    status = read_status(tmp_path)
    assert status["state"] == "completed"
    assert status["epoch_progress"] == 1.0
    assert status["overall_progress"] == 1.0
