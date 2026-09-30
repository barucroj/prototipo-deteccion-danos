"""Watch a training run from another terminal.

Usage:
    python -m src.detection.common.watch <output_dir> [--interval 5] [--once]

Reads the ``status.json`` and ``metrics.json`` the trainer keeps up to date in
the run's output folder (see :mod:`src.detection.common.audit`) and shows the
current epoch, batch, progress bars, losses, ETAs, the best epoch so far and
the per-epoch history. It only reads files, so it can be started, stopped and
restarted at any time without touching the training process.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from src.detection.common.audit import STATUS_FILE, format_duration

STALE_AFTER_S = 120


def _bar(fraction: float, width: int = 32) -> str:
    fraction = max(0.0, min(1.0, fraction or 0.0))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "." * (width - filled) + f"] {fraction * 100:5.1f}%"


def _fmt(value, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _epoch_scores(record: dict) -> tuple:
    coco = record.get("coco") or {}
    box = (coco.get("bbox") or {}).get("AP", record.get("val_map"))
    mask = (coco.get("segm") or {}).get("AP")
    return box, mask


def render(status: dict, history=None, now: float = None) -> str:
    """Human-readable view of a ``status.json`` dict (plus optional history)."""
    now = time.time() if now is None else now
    lines = []
    state = status.get("state", "?")
    phase = {"train": "TRAINING", "eval": "EVALUATING", "epoch_done": "SAVING",
             "starting": "STARTING", "done": "DONE"}.get(status.get("phase"), status.get("phase"))

    lines.append(f"{status.get('detector') or 'run'} | state: {state.upper()} | phase: {phase}")
    lines.append(f"Epoch {status.get('epoch', 0)}/{status.get('total_epochs', '?')}"
                 f"  |  batch {status.get('batch', 0)}/{status.get('total_batches', 0)}")
    lines.append(f"  epoch {_bar(status.get('epoch_progress'))}")
    lines.append(f"  total {_bar(status.get('overall_progress'))}")

    lr = status.get("lr")
    lines.append(f"Loss: last {_fmt(status.get('loss_last'))}  |  epoch mean "
                 f"{_fmt(status.get('loss_epoch_mean'))}  |  LR {lr if lr is not None else '-'}")
    components = status.get("loss_components_epoch_mean") or {}
    if components:
        lines.append("  " + "  ".join(f"{k.replace('loss_', '')} {v:.4f}"
                                      for k, v in sorted(components.items())))
    if status.get("non_finite_losses"):
        lines.append(f"  WARNING: {status['non_finite_losses']} non-finite (NaN/inf) losses so far")

    finish = status.get("expected_finish")
    lines.append(f"Elapsed {format_duration(status.get('elapsed_s'))}  |  ETA epoch "
                 f"{format_duration(status.get('eta_epoch_s'))}  |  ETA total "
                 f"{format_duration(status.get('eta_total_s'))}"
                 + (f"  (~{finish.replace('T', ' ')})" if finish and state == "running" else ""))
    if status.get("gpu_peak_memory_gb") is not None:
        lines.append(f"GPU peak memory this epoch: {status['gpu_peak_memory_gb']:.2f} GB")

    best = status.get("best")
    if best:
        lines.append(f"Best so far: epoch {best['epoch']}  ({best['select_by']} AP "
                     f"{_fmt(best['value'])})")

    updated = status.get("updated_at_unix")
    if state == "running" and updated is not None and now - updated > STALE_AFTER_S:
        lines.append(f"WARNING: no update for {format_duration(now - updated)} — is the "
                     f"process (pid {status.get('pid')}) still alive?")
    if status.get("error"):
        lines.append("ERROR: " + status["error"].strip().splitlines()[-1])

    if history:
        lines.append("")
        lines.append(f"{'epoch':>5} {'loss':>8} {'box AP':>8} {'mask AP':>8} {'time':>9}")
        for record in history:
            box, mask = _epoch_scores(record)
            best_mark = " *" if best and record.get("epoch") == best.get("epoch") else ""
            lines.append(f"{record.get('epoch', '?'):>5} {_fmt(record.get('train_loss')):>8} "
                         f"{_fmt(box):>8} {_fmt(mask):>8} "
                         f"{format_duration(record.get('epoch_time_s')):>9}{best_mark}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Watch a running (or finished) training run.")
    parser.add_argument("output_dir", help="The run's --output folder.")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between refreshes.")
    parser.add_argument("--once", action="store_true", help="Print once and exit.")
    args = parser.parse_args(argv)

    status_path = os.path.join(args.output_dir, STATUS_FILE)
    metrics_path = os.path.join(args.output_dir, "metrics.json")

    while True:
        status = _read_json(status_path)
        text = (render(status, _read_json(metrics_path)) if status
                else f"Waiting for {status_path} ...")
        if not args.once:
            os.system("cls" if os.name == "nt" else "clear")
        print(text, flush=True)
        if args.once or (status and status.get("state") != "running"):
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
