"""Training curves of a detector from its ``metrics.json`` (one entry per epoch).

Every value comes from what the trainer saved; nothing is recomputed. The COCO
numbers are on the validation split, so these figures describe training, not
the final verdict (that is COCO AP on test).

Supported files:

* current trainer (``car_parts/v2``, ``damage/v2``): total loss and its terms,
  LR, ``is_best`` and the ``coco`` block (``bbox`` and ``segm``);
* ``damage/v1``: total loss and ``coco``, without loss terms, LR or
  ``is_best`` (the best epoch is the max AP of ``select_by``).

Files without a ``coco`` block (``car_parts/v1``, homemade mAP@0.5) are rejected:
their numbers are not comparable with COCO AP.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.plots import (DPI, GRID, INK, INK_SECONDARY, MARKERS, SERIES,  # noqa: E402
                                  SURFACE)

IOU_TYPES = ("bbox", "segm")
IOU_LABELS = {"bbox": "Caja", "segm": "Máscara"}
LOSS_LABELS = {
    "loss_classifier": "Clasificación",
    "loss_box_reg": "Regresión de caja",
    "loss_mask": "Máscara",
    "loss_objectness": "Objectness (RPN)",
    "loss_rpn_box_reg": "Regresión de caja (RPN)",
}
EXTRA_SERIES = ["#8a5cd6", "#d64f8a"]
PLATEAU_TOLERANCE = 0.02  # AP points below the max that still count as the plateau
PLATEAU_MIN_EPOCHS = 3
PLATEAU_FILL = "#ecebe7"
WIDE = (7.0, 4.2)
DOUBLE = (10.0, 4.0)


@dataclass
class TrainingRun:
    """Per-epoch metrics of one trained model."""

    name: str
    epochs: List[dict]
    select_by: str

    @property
    def x(self) -> List[int]:
        return [e["epoch"] for e in self.epochs]

    @property
    def has_loss_terms(self) -> bool:
        return all("loss_components" in e for e in self.epochs)

    @property
    def has_lr(self) -> bool:
        return all("lr" in e for e in self.epochs)

    @property
    def iou_types(self) -> List[str]:
        return [t for t in IOU_TYPES if all(t in e["coco"] for e in self.epochs)]

    def coco(self, iou_type: str, key: str) -> List[Optional[float]]:
        return [e["coco"][iou_type].get(key) for e in self.epochs]

    @property
    def best_epoch(self) -> int:
        """Epoch saved as ``best_model.pth``: last ``is_best``, else max AP of ``select_by``."""
        flagged = [e["epoch"] for e in self.epochs if e.get("is_best")]
        if flagged:
            return flagged[-1]
        return max(self.epochs, key=lambda e: e["coco"][self.select_by]["AP"])["epoch"]

    def epoch(self, number: int) -> dict:
        return next(e for e in self.epochs if e["epoch"] == number)

    def plateau_start(self, tolerance: float = PLATEAU_TOLERANCE, min_epochs: int = PLATEAU_MIN_EPOCHS
                      ) -> Optional[int]:
        """First epoch from which the AP of ``select_by`` stays within ``tolerance`` of its max.

        From there on the validation AP only oscillates: which epoch is the
        "best" one is mostly noise. ``None`` if that stretch is shorter than
        ``min_epochs`` (the curve was still climbing at the end).
        """
        aps = self.coco(self.select_by, "AP")
        floor = max(aps) - tolerance
        start = len(aps)
        while start > 0 and aps[start - 1] >= floor:
            start -= 1
        if len(aps) - start < min_epochs:
            return None
        return self.epochs[start]["epoch"]


def load_run(path: str, name: str, select_by: Optional[str] = None) -> TrainingRun:
    """Read ``metrics.json`` (or the checkpoint folder holding it).

    ``select_by`` defaults to the value saved per epoch; files that do not save
    it (``damage/v1``) need it explicitly.
    """
    if os.path.isdir(path):
        path = os.path.join(path, "metrics.json")
    with open(path, encoding="utf-8") as f:
        epochs = json.load(f)
    if not epochs:
        raise ValueError(f"{path}: sin épocas")
    if not all("coco" in e for e in epochs):
        raise ValueError(f"{path}: sin bloque 'coco' (métrica casera, no comparable con COCO AP)")
    epochs = sorted(epochs, key=lambda e: e["epoch"])
    select_by = select_by or epochs[-1].get("select_by")
    if select_by not in IOU_TYPES:
        raise ValueError(f"{path}: indique select_by ('bbox' o 'segm'); el archivo no lo guarda")
    return TrainingRun(name=name, epochs=epochs, select_by=select_by)


# ---------------------------------------------------------------------------
# Drawing helpers (same look as the sweep figures in src/evaluation/plots.py)
# ---------------------------------------------------------------------------

def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.xaxis.get_major_locator().set_params(integer=True)


def _figure(title: str, columns: int = 1):
    fig, axes = plt.subplots(1, columns, figsize=WIDE if columns == 1 else DOUBLE, dpi=DPI, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    return fig, list(axes[0])


def _color(slot: int) -> str:
    palette = SERIES + EXTRA_SERIES
    return palette[slot % len(palette)]


def _line(ax, xs, ys, slot: int, label: str, color: Optional[str] = None, linestyle: str = "-"):
    points = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not points:
        return
    px, py = zip(*points)
    ax.plot(px, py, color=color or _color(slot), marker=MARKERS[slot % len(MARKERS)], markersize=4,
            linewidth=2, linestyle=linestyle, label=label, markeredgecolor=SURFACE, markeredgewidth=1)


def _best(ax, run: TrainingRun):
    best = run.best_epoch
    ax.axvline(best, color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.annotate(f"mejor época {best}", (best, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -2),
                textcoords="offset points", fontsize=8, color=INK_SECONDARY, va="top")


def _legend(fig, axes, columns: int):
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=columns, fontsize=8, frameon=False)


def _save(fig, output_dir: str, name: str, legend: bool = True) -> str:
    path = os.path.join(output_dir, name)
    fig.tight_layout(rect=(0, 0.08 if legend else 0, 1, 0.95))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def _ap_axis(ax):
    ax.set_ylim(0, 1.0)


# ---------------------------------------------------------------------------
# Figures of one model
# ---------------------------------------------------------------------------

def plot_run(run: TrainingRun, output_dir: str) -> List[str]:
    """Write every figure of ``run`` as ``<name>_<figure>.png`` and return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    xs, prefix, paths = run.x, run.name, []
    iou_types = run.iou_types

    fig, (ax,) = _figure(f"{run.name}: loss de entrenamiento")
    _style(ax, "", "Época", "Loss")
    _line(ax, xs, [e["train_loss"] for e in run.epochs], 0, "Loss total")
    _best(ax, run)
    ax.set_ylim(bottom=0)
    paths.append(_save(fig, output_dir, f"{prefix}_01_loss.png", legend=False))

    if run.has_loss_terms:
        terms = list(run.epochs[0]["loss_components"])
        fig, (ax,) = _figure(f"{run.name}: componentes del loss")
        _style(ax, "", "Época", "Loss")
        for slot, term in enumerate(terms):
            _line(ax, xs, [e["loss_components"].get(term) for e in run.epochs], slot, LOSS_LABELS.get(term, term))
        _best(ax, run)
        ax.set_ylim(bottom=0)
        _legend(fig, [ax], 3)
        paths.append(_save(fig, output_dir, f"{prefix}_02_loss_componentes.png"))

    fig, axes = _figure(f"{run.name}: COCO AP en validación", len(iou_types))
    for ax, iou in zip(axes, iou_types):
        _style(ax, IOU_LABELS[iou], "Época", "AP")
        for slot, key in enumerate(("AP", "AP50", "AP75")):
            label = "AP@[.5:.95]" if key == "AP" else key
            _line(ax, xs, run.coco(iou, key), slot, label)
        _best(ax, run)
        _ap_axis(ax)
    _legend(fig, axes, 3)
    paths.append(_save(fig, output_dir, f"{prefix}_03_map.png"))

    fig, (ax,) = _figure(f"{run.name}: loss contra AP de validación")
    _style(ax, "", "Época", "Loss de entrenamiento")
    _line(ax, xs, [e["train_loss"] for e in run.epochs], 0, "Loss total")
    ax.set_ylim(bottom=0)
    right = ax.twinx()
    right.set_ylabel("AP@[.5:.95] (val)", color=INK_SECONDARY, fontsize=9)
    right.tick_params(colors=INK_SECONDARY, labelsize=8)
    for side in ("top", "left", "bottom"):
        right.spines[side].set_visible(False)
    right.spines["right"].set_color(GRID)
    for slot, iou in enumerate(iou_types, start=1):
        _line(right, xs, run.coco(iou, "AP"), slot, f"AP {IOU_LABELS[iou].lower()}")
    right.set_ylim(0, 1.0)
    plateau = run.plateau_start()
    if plateau is not None:
        ax.axvspan(plateau - 0.5, xs[-1] + 0.5, color=PLATEAU_FILL, zorder=0,
                   label=f"Meseta: AP {IOU_LABELS[run.select_by].lower()} a ≤ {PLATEAU_TOLERANCE} del máximo")
        ax.annotate(f"meseta desde la época {plateau}", (plateau - 0.5, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK_SECONDARY)
    _best(ax, run)
    handles = ax.get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + right.get_legend_handles_labels()[1]
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    paths.append(_save(fig, output_dir, f"{prefix}_04_loss_vs_map.png"))

    fig, axes = _figure(f"{run.name}: AP por tamaño de objeto (validación)", len(iou_types))
    for ax, iou in zip(axes, iou_types):
        _style(ax, IOU_LABELS[iou], "Época", "AP@[.5:.95]")
        for slot, (key, label) in enumerate((("AP_small", "Pequeño (< 32² px)"),
                                             ("AP_medium", "Mediano"), ("AP_large", "Grande (> 96² px)"))):
            _line(ax, xs, [v if v is None or v >= 0 else None for v in run.coco(iou, key)], slot, label)
        _best(ax, run)
        _ap_axis(ax)
    _legend(fig, axes, 3)
    paths.append(_save(fig, output_dir, f"{prefix}_05_map_por_tamano.png"))

    fig, axes = _figure(f"{run.name}: recall promedio (validación)", len(iou_types))
    for ax, iou in zip(axes, iou_types):
        _style(ax, IOU_LABELS[iou], "Época", "AR@[.5:.95]")
        for slot, key in enumerate(("AR_1", "AR_10", "AR_100")):
            _line(ax, xs, run.coco(iou, key), slot, f"AR@{key.split('_')[1]} detecciones")
        _best(ax, run)
        _ap_axis(ax)
    _legend(fig, axes, 3)
    paths.append(_save(fig, output_dir, f"{prefix}_06_recall.png"))

    if run.has_lr:
        fig, (ax,) = _figure(f"{run.name}: tasa de aprendizaje")
        _style(ax, "", "Época", "LR")
        _line(ax, xs, [e["lr"] for e in run.epochs], 0, "LR")
        ax.set_yscale("log")
        _best(ax, run)
        paths.append(_save(fig, output_dir, f"{prefix}_07_lr.png", legend=False))
    return paths


def plot_comparison(runs: Sequence[TrainingRun], output_dir: str, name: str, title: str) -> str:
    """AP@[.5:.95] (box and mask) of several versions of the same detector on one figure."""
    os.makedirs(output_dir, exist_ok=True)
    iou_types = [t for t in IOU_TYPES if all(t in r.iou_types for r in runs)]
    fig, axes = _figure(title, len(iou_types) + 1)
    _style(axes[0], "Loss de entrenamiento", "Época", "Loss")
    for slot, run in enumerate(runs):
        _line(axes[0], run.x, [e["train_loss"] for e in run.epochs], slot, run.name)
    axes[0].set_ylim(bottom=0)
    for ax, iou in zip(axes[1:], iou_types):
        _style(ax, f"AP {IOU_LABELS[iou].lower()} (val), ★ = mejor época", "Época", "AP@[.5:.95]")
        for slot, run in enumerate(runs):
            _line(ax, run.x, run.coco(iou, "AP"), slot, run.name)
        for slot, run in enumerate(runs):
            best = run.epoch(run.best_epoch)["coco"][iou]["AP"]
            ax.plot([run.best_epoch], [best], marker="*", markersize=13, color=_color(slot),
                    markeredgecolor=INK, markeredgewidth=0.6, linestyle="none", zorder=5)
        _ap_axis(ax)
    _legend(fig, axes, len(runs))
    return _save(fig, output_dir, f"{name}.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summary_rows(runs: Sequence[TrainingRun]) -> List[Dict]:
    """One row per model with the validation metrics of its best epoch."""
    rows = []
    for run in runs:
        best = run.epoch(run.best_epoch)
        row = {"modelo": run.name, "epocas": len(run.epochs), "mejor_epoca": run.best_epoch,
               "seleccion": run.select_by, "meseta_desde": run.plateau_start(),
               "loss_mejor_epoca": best["train_loss"]}
        for iou in run.iou_types:
            for key in ("AP", "AP50", "AP75", "AR_100"):
                row[f"{iou}_{key}"] = best["coco"][iou][key]
        rows.append(row)
    return rows


def write_summary(runs: Sequence[TrainingRun], figures: Dict[str, List[str]], output_dir: str) -> str:
    """Write ``resumen.md`` (best epoch per model and list of figures)."""
    rows = summary_rows(runs)
    lines = ["# Curvas de entrenamiento", "",
             "Métricas COCO en **validación** de la época guardada como `best_model.pth`. "
             "El veredicto de cada modelo es el AP en **test** (notebooks de evaluación), no esta tabla.", "",
             f"**Meseta**: primera época desde la que el AP de la métrica de selección se queda a "
             f"≤ {PLATEAU_TOLERANCE} de su máximo; dentro de ella, qué época sale \"mejor\" es sobre todo ruido.", "",
             "| Modelo | Épocas | Mejor época | Selección | Meseta desde | Loss | AP caja | AP50 caja | AP75 caja | AR@100 caja "
             "| AP máscara | AP50 máscara | AP75 máscara | AR@100 máscara |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
    for r in rows:
        cells = [r["modelo"], str(r["epocas"]), str(r["mejor_epoca"]), r["seleccion"],
                 "—" if r["meseta_desde"] is None else str(r["meseta_desde"]), fmt(r["loss_mejor_epoca"])]
        cells += [fmt(r.get(f"{iou}_{k}")) for iou in IOU_TYPES for k in ("AP", "AP50", "AP75", "AR_100")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Figuras", ""]
    for group, paths in figures.items():
        lines.append(f"- **{group}**: " + ", ".join(f"`{os.path.basename(p)}`" for p in paths))
    path = os.path.join(output_dir, "resumen.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(output_dir, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return path
