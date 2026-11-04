"""Figures of the damage-threshold sweep, as PNG for the report.

Colors are the first categorical slots of a palette validated for color-vision
deficiency; two of them sit below 3:1 contrast on white, so every series also
has its own marker and a direct label at its end.
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
FIGURE_SIZE = (7.0, 4.2)
DPI = 200


def _axes(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    return fig, ax


def _line(ax, xs, ys, slot: int, label: str):
    points = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not points:
        return
    px, py = zip(*points)
    ax.plot(px, py, color=SERIES[slot], marker=MARKERS[slot], markersize=5, linewidth=2, label=label,
            markeredgecolor=SURFACE, markeredgewidth=1)
    ax.annotate(label, (px[-1], py[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=8, color=INK_SECONDARY)


def _best_line(ax, best: Optional[dict]):
    if best:
        ax.axvline(best["umbral_danos"], color=INK_SECONDARY, linestyle="--", linewidth=1)
        ax.annotate(f"mejor {best['umbral_danos']:.2f}", (best["umbral_danos"], 1.0), xytext=(4, -2),
                    textcoords="offset points", fontsize=8, color=INK_SECONDARY, va="top")


def _legend(ax, columns: int):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=columns, fontsize=8, frameon=False)


def _save(fig, output_dir: str, name: str) -> str:
    path = os.path.join(output_dir, name)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_sweep(rows: List[dict], best: Optional[dict], main_iou: float, n_pairs: int, output_dir: str) -> List[str]:
    """Write the sweep figures in ``output_dir`` and return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    main = [r for r in rows if r["min_iou"] == main_iou]
    xs = [r["umbral_danos"] for r in main]
    span = (min(xs) - 0.02, max(xs) + 0.09)
    paths = []

    fig, ax = _axes(f"Daños nuevos según el umbral de confianza (IoU ≥ {main_iou})",
                    "Umbral de confianza de daños (M3)", "Valor")
    _line(ax, xs, [r["precision"] for r in main], 0, "Precision")
    _line(ax, xs, [r["recall"] for r in main], 1, "Recall")
    _line(ax, xs, [r["f1"] for r in main], 2, "F1")
    _line(ax, xs, [r["pares_exactos"] / n_pairs if n_pairs else None for r in main], 3, "Pares exactos")
    _best_line(ax, best)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(*span)
    _legend(ax, 4)
    paths.append(_save(fig, output_dir, "prf_vs_umbral.png"))

    fig, ax = _axes(f"Aciertos y errores en daños nuevos (IoU ≥ {main_iou})",
                    "Umbral de confianza de daños (M3)", "Número de daños")
    _line(ax, xs, [r["tp"] for r in main], 0, "Nuevos acertados (TP)")
    _line(ax, xs, [r["fp"] for r in main], 1, "Nuevos falsos (FP)")
    _line(ax, xs, [r["fn"] for r in main], 2, "Nuevos perdidos (FN)")
    if best:
        ax.axvline(best["umbral_danos"], color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.set_ylim(bottom=0)
    ax.set_xlim(min(xs) - 0.02, max(xs) + 0.14)
    ax.yaxis.get_major_locator().set_params(integer=True)
    _legend(ax, 3)
    paths.append(_save(fig, output_dir, "errores_vs_umbral.png"))

    fig, ax = _axes(f"Precision contra recall de daños nuevos (IoU ≥ {main_iou})", "Recall", "Precision")
    points = [r for r in main if r["precision"] is not None and r["recall"] is not None]
    ax.plot([r["recall"] for r in points], [r["precision"] for r in points], color=SERIES[0], marker="o",
            markersize=5, linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1)
    for i, r in enumerate(points):
        is_best = best is not None and r["umbral_danos"] == best["umbral_danos"]
        if is_best or i in (0, len(points) - 1):
            ax.annotate(f"{r['umbral_danos']:.2f}" + (" (mejor)" if is_best else ""), (r["recall"], r["precision"]),
                        xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK_SECONDARY)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    paths.append(_save(fig, output_dir, "precision_recall.png"))

    ious = sorted({r["min_iou"] for r in rows})
    if len(ious) > 1:
        fig, ax = _axes("F1 de daños nuevos según el IoU exigido", "Umbral de confianza de daños (M3)", "F1")
        for slot, iou in enumerate(ious[:len(SERIES)]):
            subset = [r for r in rows if r["min_iou"] == iou]
            _line(ax, [r["umbral_danos"] for r in subset], [r["f1"] for r in subset], slot, f"IoU ≥ {iou}")
        _best_line(ax, best)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(*span)
        _legend(ax, len(ious))
        paths.append(_save(fig, output_dir, "f1_por_iou.png"))
    return paths
