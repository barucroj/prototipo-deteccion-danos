"""Run auditing for the shared trainer: a live status file, a log and a run record.

A training run can take hours and is usually launched in the background, so
the console alone is not enough to know where it is. Every run writes three
extra files next to its checkpoints:

- ``run_info.json`` — written once at start: CLI args, dataset config, git
  commit (and whether the tree was dirty), library versions, GPU, seed,
  dataset sizes and per-class instance counts. Enough to reproduce or
  explain the run later.
- ``status.json`` — rewritten every few seconds: phase (train/eval), epoch,
  batch, % of the epoch and of the whole run, current/mean loss per
  component, LR, GPU memory, elapsed time, ETAs and the best epoch so far.
  Read it with ``python -m src.detection.common.watch <output_dir>``.
- ``train.log`` — timestamped copy of every message the trainer prints
  (tqdm bars stay console-only), plus the final state or the traceback.

``status.json`` is written atomically (temp file + ``os.replace``) so a reader
never sees half a file; if a reader happens to hold it open on Windows the
write is skipped and retried on the next update instead of failing training.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime

import torch

STATUS_FILE = "status.json"
RUN_INFO_FILE = "run_info.json"
LOG_FILE = "train.log"


def format_duration(seconds) -> str:
    """``3725 -> "1:02:05"``; ``None`` -> ``"--:--:--"``."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _write_json_atomic(path: str, data) -> bool:
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
        return True
    except PermissionError:
        # A reader has the file open (Windows); the next update will catch up.
        return False


def git_info(repo_dir: str = ".") -> dict:
    """Current commit and whether tracked files have uncommitted changes."""
    def run(*cmd):
        return subprocess.run(["git", *cmd], cwd=repo_dir, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    try:
        return {
            "commit": run("rev-parse", "HEAD") or None,
            "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "branch": None, "dirty": None}


def environment_info(device) -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except ImportError:  # pragma: no cover
        pass
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.device(device))
        info["cuda"] = torch.version.cuda
        info["gpu"] = props.name
        info["gpu_total_memory_gb"] = round(props.total_memory / 1024 ** 3, 2)
    return info


def dataset_summary(dataset) -> dict:
    """Image count, instance count and per-class instance counts of a split."""
    counts = Counter()
    for sample in dataset.samples:
        for ann in sample[4]:
            counts[ann["category_id"]] += 1
    return {
        "images": len(dataset),
        "instances": sum(counts.values()),
        "instances_per_class": {
            dataset.categories.get(cid, str(cid)): n for cid, n in sorted(counts.items())
        },
        "classes_without_instances": sorted(
            name for cid, name in dataset.categories.items() if counts[cid] == 0
        ),
    }


class RunAuditor:
    """Tracks one training run and mirrors its state to disk.

    The trainer calls :meth:`start_epoch`, then :meth:`start_phase` for
    ``"train"`` and (on evaluation epochs) ``"eval"``, :meth:`on_batch` from
    inside each loop, :meth:`end_epoch` with the epoch record, and finally
    :meth:`finish`.

    Args:
        output_dir: The run's checkpoint folder.
        total_epochs: Epochs the run was asked for.
        update_every_s: Minimum seconds between ``status.json`` rewrites
            during a loop (phase changes and epoch ends always write).
        clock: Injectable time source, for tests.
    """

    def __init__(self, output_dir: str, total_epochs: int, update_every_s: float = 2.0,
                 clock=time.time):
        self.output_dir = output_dir
        self.total_epochs = total_epochs
        self.update_every_s = update_every_s
        self.clock = clock
        os.makedirs(output_dir, exist_ok=True)

        self.status_path = os.path.join(output_dir, STATUS_FILE)
        self.log_path = os.path.join(output_dir, LOG_FILE)
        self.start_time = clock()
        self._last_write = 0.0

        self.detector = None
        self.select_by = None
        self.epoch = 0
        self.eval_this_epoch = False
        self.phase = "starting"
        self.batch = 0
        self.total_batches = 0
        self.phase_start = self.start_time
        self.epoch_start = self.start_time
        self.n_train_batches = 0
        self.n_val_batches = 0

        self.loss_sum = 0.0
        self.loss_count = 0
        self.last_loss = None
        self.component_sums = defaultdict(float)
        self.non_finite_losses = 0

        self.epoch_times = []
        self.best = None
        self.last_epoch = None
        self.optimizer = None
        self.device = None

    # ---------------------------------------------------------------- logging
    def log(self, message: str = "") -> None:
        """Print ``message`` and append it, timestamped, to ``train.log``."""
        print(message)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            for line in str(message).splitlines() or [""]:
                f.write(f"[{stamp}] {line}\n")

    def write_run_info(self, info: dict) -> None:
        info = dict(info)
        info.setdefault("started_at", datetime.fromtimestamp(self.start_time).isoformat())
        _write_json_atomic(os.path.join(self.output_dir, RUN_INFO_FILE), info)

    # --------------------------------------------------------------- tracking
    def configure(self, detector: str, select_by: str, n_train_batches: int,
                  n_val_batches: int, optimizer=None, device=None) -> None:
        self.detector = detector
        self.select_by = select_by
        self.n_train_batches = n_train_batches
        self.n_val_batches = n_val_batches
        self.optimizer = optimizer
        self.device = torch.device(device) if device is not None else None

    def start_epoch(self, epoch: int, will_eval: bool) -> None:
        self.epoch = epoch
        self.eval_this_epoch = will_eval
        self.epoch_start = self.clock()
        self.loss_sum = 0.0
        self.loss_count = 0
        self.last_loss = None
        self.component_sums = defaultdict(float)
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def start_phase(self, phase: str, total_batches: int) -> None:
        self.phase = phase
        self.batch = 0
        self.total_batches = total_batches
        self.phase_start = self.clock()
        self.write_status(force=True)

    def on_batch(self, batch_index: int, loss=None, loss_components=None) -> None:
        """Called after each batch; ``batch_index`` is 0-based."""
        self.batch = batch_index + 1
        if loss is not None:
            if math.isfinite(loss):
                self.loss_sum += loss
                self.loss_count += 1
            else:
                self.non_finite_losses += 1
                self.log(f"  WARNING: non-finite loss ({loss}) at epoch {self.epoch}, "
                         f"batch {self.batch}")
            self.last_loss = loss
            for name, value in (loss_components or {}).items():
                self.component_sums[name] += value
        self.write_status()

    def end_epoch(self, record: dict, is_best: bool, selected_value) -> None:
        self.epoch_times.append(self.clock() - self.epoch_start)
        self.last_epoch = record
        if is_best:
            self.best = {"epoch": record["epoch"], "select_by": self.select_by,
                         "value": selected_value}
        self.phase = "epoch_done"
        self.write_status(force=True)

    def epoch_loss_components(self) -> dict:
        return {k: v / self.loss_count for k, v in self.component_sums.items()} \
            if self.loss_count else {}

    def peak_gpu_memory_gb(self):
        if self.device is not None and self.device.type == "cuda":
            return round(torch.cuda.max_memory_allocated(self.device) / 1024 ** 3, 3)
        return None

    # --------------------------------------------------------------- progress
    def _train_share(self) -> float:
        """Fraction of this epoch's batches that are training (vs. evaluation)."""
        if not self.eval_this_epoch:
            return 1.0
        total = self.n_train_batches + self.n_val_batches
        return self.n_train_batches / total if total else 1.0

    def epoch_fraction(self) -> float:
        if self.phase in ("epoch_done", "done"):
            return 1.0
        if self.phase not in ("train", "eval") or not self.total_batches:
            return 0.0
        done = self.batch / self.total_batches
        share = self._train_share()
        return done * share if self.phase == "train" else share + done * (1.0 - share)

    def overall_fraction(self) -> float:
        if not self.total_epochs or self.epoch == 0:
            return 0.0
        return min(1.0, (self.epoch - 1 + self.epoch_fraction()) / self.total_epochs)

    def eta_epoch_s(self):
        """Seconds left in the current epoch, from the current phase's batch rate."""
        if self.phase in ("epoch_done", "done"):
            return 0.0
        if self.phase not in ("train", "eval") or self.batch == 0:
            return self.epoch_times[-1] if self.epoch_times else None
        rate = (self.clock() - self.phase_start) / self.batch
        left = (self.total_batches - self.batch) * rate
        if self.phase == "train" and self.eval_this_epoch:
            # Eval runs at roughly training speed or faster; the rate is an upper bound.
            left += self.n_val_batches * rate
        return left

    def eta_total_s(self):
        epoch_left = self.eta_epoch_s()
        if epoch_left is None:
            return None
        remaining_epochs = self.total_epochs - self.epoch
        if remaining_epochs <= 0:
            return epoch_left
        if self.epoch_times:
            per_epoch = sum(self.epoch_times) / len(self.epoch_times)
        else:
            frac = self.epoch_fraction()
            elapsed = self.clock() - self.epoch_start
            per_epoch = elapsed / frac if frac > 0 else None
        return None if per_epoch is None else epoch_left + remaining_epochs * per_epoch

    # ----------------------------------------------------------------- output
    def status(self, state: str = "running", error: str = None) -> dict:
        now = self.clock()
        eta_total = self.eta_total_s() if state == "running" else 0.0
        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer is not None else None
        status = {
            "state": state,
            "detector": self.detector,
            "phase": self.phase,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "batch": self.batch,
            "total_batches": self.total_batches,
            "epoch_progress": round(self.epoch_fraction(), 4),
            "overall_progress": round(self.overall_fraction(), 4),
            "loss_last": self.last_loss,
            "loss_epoch_mean": self.loss_sum / self.loss_count if self.loss_count else None,
            "loss_components_epoch_mean": self.epoch_loss_components(),
            "non_finite_losses": self.non_finite_losses,
            "lr": lr,
            "gpu_peak_memory_gb": self.peak_gpu_memory_gb(),
            "elapsed_s": round(now - self.start_time, 1),
            "eta_epoch_s": self.eta_epoch_s() if state == "running" else 0.0,
            "eta_total_s": eta_total,
            "expected_finish": (datetime.fromtimestamp(now + eta_total).isoformat(timespec="seconds")
                                if eta_total is not None else None),
            "epoch_times_s": [round(t, 1) for t in self.epoch_times],
            "best": self.best,
            "last_epoch": self.last_epoch,
            "pid": os.getpid(),
            "updated_at": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "updated_at_unix": now,
        }
        if error:
            status["error"] = error
        return status

    def write_status(self, force: bool = False, state: str = "running", error: str = None):
        now = self.clock()
        if not force and now - self._last_write < self.update_every_s:
            return
        if _write_json_atomic(self.status_path, self.status(state, error)):
            self._last_write = now

    def finish(self, state: str, error: str = None) -> None:
        """``state`` is ``"completed"``, ``"interrupted"`` or ``"failed"``."""
        if state == "completed":
            self.phase = "done"
        self.write_status(force=True, state=state, error=error)
        self.log(f"Run {state}. Total time {format_duration(self.clock() - self.start_time)}.")
        if error:
            self.log(error)
