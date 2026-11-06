"""Adoption rule of ``car_parts/v3_mix`` against ``car_parts/v3_10c`` (decided 2026-11-06).

Usage:
    python -m src.m2_parts.decide_mix --roboflow DIR/compare_10c_roboflow_test.json
        --hitl DIR/compare_10c_hitl_test.json [--candidate v3_mix] [--reference v3_10c]
        [--iou segm] [--output DIR]

Both reports come from :mod:`src.m2_parts.compare_10c` on the **test** split,
with ``--reference`` set to the reference model. ``v3_mix`` is adopted when,
on the chosen AP (mask AP@0.5:0.95 by default, the metric that selects the
epoch and the output M3 consumes):

1. it does **not** get distinguishably worse on the Roboflow test: the 95 %
   interval of ``candidate - reference`` is not entirely below zero. With 16
   images the interval is wide, so this is non-inferiority only, never an
   improvement;
2. it **improves** on the HITL test (the interval is entirely above zero) —
   **only when** ``hitl_aligned`` (``--hitl-aligned``) says the HITL labeling
   criterion is aligned with Roboflow (updated 2026-11-06). Until then the
   decision rests on rule 1 alone and HITL is reported apart.

The other AP (box) is reported alongside, without deciding. The HITL test may
be optimistic (:data:`src.m2_parts.compare_10c.HITL_WARNING`).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

from src.m2_parts.compare_10c import HITL_WARNING

IOU_NAMES = {"bbox": "caja", "segm": "máscara"}


def _difference(report: dict, dataset: str, candidate: str, reference: str, iou: str) -> dict:
    if report.get("dataset") != dataset:
        raise ValueError(f"expected a {dataset!r} report, got {report.get('dataset')!r}")
    if report.get("split") != "test":
        raise ValueError(f"the {dataset} report is on split {report.get('split')!r}; the rule needs 'test'")
    if report.get("reference") != reference:
        raise ValueError(f"the {dataset} report uses reference {report.get('reference')!r}, not {reference!r}")
    try:
        return report["differences_vs_reference"][candidate][iou]["AP"]
    except KeyError:
        raise ValueError(f"the {dataset} report has no {iou} difference for {candidate!r}") from None


def not_worse(diff: dict) -> Optional[bool]:
    """The interval is not entirely below zero (``None`` without an interval)."""
    high = diff["ci95"]["high"]
    return None if high is None else high >= 0


def improves(diff: dict) -> Optional[bool]:
    """The interval is entirely above zero (``None`` without an interval)."""
    low = diff["ci95"]["low"]
    return None if low is None else low > 0


NOT_ALIGNED = ("El criterio de etiqueta de HITL todavia no esta alineado con Roboflow: el test de "
               "HITL se reporta aparte y no cuenta para la decision.")


def decide(roboflow: dict, hitl: dict, candidate: str = "v3_mix", reference: str = "v3_10c",
           iou: str = "segm", hitl_aligned: bool = False) -> dict:
    """Apply the rule on ``iou`` and report the other AP too."""
    out = {"candidate": candidate, "reference": reference, "decides_on": iou, "hitl_aligned": hitl_aligned,
           "warning": HITL_WARNING if hitl_aligned else f"{NOT_ALIGNED} {HITL_WARNING}", "by_iou": {}}
    for kind in ("segm", "bbox"):
        try:
            r = _difference(roboflow, "roboflow", candidate, reference, kind)
            h = _difference(hitl, "hitl", candidate, reference, kind)
        except ValueError:
            if kind == iou:
                raise
            continue
        ok_r, ok_h = not_worse(r), improves(h)
        out["by_iou"][kind] = {
            "roboflow": {"difference": r["value"], "ci95": r["ci95"], "no_empeora": ok_r},
            "hitl": {"difference": h["value"], "ci95": h["ci95"], "mejora": ok_h},
            "cumple": bool(ok_r and (ok_h or not hitl_aligned)),
        }
    out["adopt"] = out["by_iou"][iou]["cumple"]
    return out


def _ci(d):
    ci = d["ci95"]
    if d["difference"] is None or ci["low"] is None:
        return "-"
    return f"{d['difference']:+.4f} [{ci['low']:+.4f}, {ci['high']:+.4f}]"


def render_markdown(decision: dict) -> str:
    c, r = decision["candidate"], decision["reference"]
    hitl_rule = ("**y** en el test de HITL el IC queda por encima de cero." if decision["hitl_aligned"] else
                 "El test de HITL se reporta aparte y **no cuenta** (criterio sin alinear).")
    lines = [f"# ¿Se adopta {c}? — {'SÍ' if decision['adopt'] else 'NO'}", "",
             f"Regla (AP de {IOU_NAMES[decision['decides_on']]} @0.5:0.95, {c} − {r}, IC 95 % "
             "bootstrap pareado por imagen): en el test de Roboflow el IC no queda por debajo de cero "
             f"(no inferioridad; con 16 imágenes no cuenta como mejora). {hitl_rule}", "",
             f"> {decision['warning']}", "",
             "| AP | test de Roboflow | ¿no empeora? | test de HITL | ¿mejora? | cumple |",
             "|---|---|---|---|---|---|"]
    for kind, d in decision["by_iou"].items():
        mark = " (decide)" if kind == decision["decides_on"] else ""
        yes = {True: "sí", False: "no", None: "sin datos"}
        lines.append(f"| {IOU_NAMES[kind]}{mark} | {_ci(d['roboflow'])} | {yes[d['roboflow']['no_empeora']]} | "
                     f"{_ci(d['hitl'])} | {yes[d['hitl']['mejora']]} | {'sí' if d['cumple'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Apply the v3_mix adoption rule to two compare_10c reports.")
    p.add_argument("--roboflow", required=True, help="compare_10c_roboflow_test.json")
    p.add_argument("--hitl", required=True, help="compare_10c_hitl_test.json")
    p.add_argument("--candidate", default="v3_mix")
    p.add_argument("--reference", default="v3_10c")
    p.add_argument("--iou", choices=("segm", "bbox"), default="segm")
    p.add_argument("--hitl-aligned", action="store_true",
                   help="Count the HITL test (only once its labeling criterion is aligned with Roboflow).")
    p.add_argument("--output", default=None, help="Folder for decision_v3_mix.json/.md (default: the Roboflow report's).")
    args = p.parse_args(argv)
    with open(args.roboflow, encoding="utf-8") as f:
        roboflow = json.load(f)
    with open(args.hitl, encoding="utf-8") as f:
        hitl = json.load(f)
    decision = decide(roboflow, hitl, args.candidate, args.reference, args.iou, args.hitl_aligned)
    output = args.output or os.path.dirname(os.path.abspath(args.roboflow))
    os.makedirs(output, exist_ok=True)
    stem = os.path.join(output, f"decision_{args.candidate}")
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)
    text = render_markdown(decision)
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return decision


if __name__ == "__main__":
    main()
