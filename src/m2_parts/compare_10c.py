"""Compare car_parts models on the ten final classes (``car_parts/v3_10c`` plan).

Usage:
    python -m src.m2_parts.compare_10c --model v2=V2.pth --model v3=V3.pth
        --reference v2 --split test --output DIR
        [--dataset roboflow|hitl|mix|carparts_seg] [--bootstrap 1000] [--seed 0] [--nms-iou 0.5]

Every model is scored on the same images and the same ten classes
(:data:`src.m2_parts.class_map_10.FINAL_CLASSES`):

- a model trained on the ten classes (Roboflow or Carparts-Seg map) is scored
  as is;
- ``car_parts/v2`` (29 classes) has its predictions mapped through
  :data:`~src.m2_parts.class_map_10.MAP_V2_TO_10`, with NMS within each final
  class so two merged predictions (front and back door) do not count twice.
  light is headlights and taillights only: v2's ``fog_light`` and
  ``indicator_light`` predictions are dropped, as are those annotations.

For each model: box and mask AP@0.5:0.95 and AP50 with a 95% interval, AP of
each class with its instance count (*poco fiable* under 10) and interval, and
AP of two groups: ``carparts_seg_sin_etiqueta`` (fender, grille, glass:
Carparts-Seg labels the first two not at all and glass only as windshield and
rear window) and ``resto``. Each other model is compared with ``--reference``:
difference and paired bootstrap interval (same resampled images for every
model). The verdict is *mejora* / *empeora* only when the interval excludes
zero; otherwise *no distinguible*.

With ``--dataset roboflow``, any v2 model is also scored on its own 29
classes, apart, as historical data (not comparable with the ten-class
numbers).

``--dataset hitl`` scores the Humans in the Loop test split and ``--dataset
mix`` the Roboflow and HITL test splits together (:data:`HITL_WARNING`: SIFT
cannot rule out photos of the same vehicle across splits, so it may be
optimistic). :mod:`src.m2_parts.decide_mix` applies the adoption rule of
``v3_mix`` to the Roboflow and HITL reports.

``--dataset carparts_seg`` is an **external test** of models trained only on
Roboflow: the Carparts-Seg test split, restricted to the classes labeled with
the same criterion as Roboflow (:data:`EXTERNAL_CLASSES`: light, bumper, hood,
wheel, mirror, trunk; not door, whose mask includes the window there, nor
glass, labeled only as windshield and rear window). Its images come from 579
base photos with augmented copies, so the interval is optimistic, and it is
never used to choose a model. Carparts-Seg does not label fog lights either,
so a fog light predicted as light counts as a false positive there.

Reuses :func:`src.m2_parts.compare_versions.per_image_eval` and
:func:`~src.m2_parts.compare_versions.accumulate` (exact against pycocotools,
with per-image weights for the bootstrap).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.detection_common.coco_dataset import build_dataset, collate_fn, final_categories
from src.detection_common.coco_eval import collect_detections
from src.m2_parts.class_map import CLASS_MAP
from src.m2_parts.class_map_10 import CLASS_MAP_ROBOFLOW, FINAL_CLASSES, MAP_V2_TO_10
from src.m2_parts.compare_versions import (
    UNRELIABLE_BELOW, accumulate, instance_counts, interval, load_version, per_image_eval,
    to_final_detections,
)
from src.m2_parts.config import (CARPARTS_SEG_10_CONFIG, HITL_10_CONFIG, MASK_CONFIG, MIX_10_DATA,
                                  ROBOFLOW_10_CONFIG)

GROUPS = {"carparts_seg_sin_etiqueta": ("fender", "grille", "glass"),
          "resto": tuple(c for c in FINAL_CLASSES if c not in ("fender", "grille", "glass"))}
DATASETS = {"roboflow": ROBOFLOW_10_CONFIG, "hitl": HITL_10_CONFIG, "mix": MIX_10_DATA,
            "carparts_seg": CARPARTS_SEG_10_CONFIG}
HITL_WARNING = ("Test de HITL: puede ser optimista. Las particiones agrupan las fotos del mismo "
                "vehiculo que SIFT encontro (17 pares, 15 a 43 puntos), pero por debajo de 15 puntos "
                "SIFT no distingue el mismo coche de una coincidencia casual: puede haber fotos del "
                "mismo vehiculo en train y en test.")
#: Classes scored in the Carparts-Seg external test (same labeling criterion as Roboflow).
EXTERNAL_CLASSES = ("light", "bumper", "hood", "wheel", "mirror", "trunk")
EXTERNAL_GROUPS = {"criterio_compartido": EXTERNAL_CLASSES}
EXTERNAL_WARNING = ("PRUEBA EXTERNA: test de Carparts-Seg solo en light, bumper, hood, wheel, mirror "
                    "y trunk (door y glass tienen otro criterio de etiqueta). Sus imagenes vienen de "
                    "579 fotos base con copias aumentadas y se parecen entre si: el IC es optimista. "
                    "No se usa para elegir modelo. Carparts-Seg no etiqueta faros de niebla. wheel: 35 de "
                    "115 anotaciones del test unen las dos ruedas en un solo poligono (artefacto del "
                    "formato), y en Roboflow wheel es el rin (la rueda completa es tire): su AP aqui no "
                    "es interpretable.")
VAL_WARNING = ("WARNING: the validation split chooses each model's best checkpoint; "
               "the verdict must be computed on 'test'.")


# ------------------------------------------------------------------------ class mapping
def model_kind(categories: Dict[int, str], class_map) -> str:
    """``"ten"`` for a model on the ten classes, ``"v2"`` for one on v2's 29 classes."""
    names = set(categories.values())
    if names <= set(FINAL_CLASSES):
        return "ten"
    if class_map == CLASS_MAP or names == set(MAP_V2_TO_10):
        return "v2"
    raise ValueError(f"model classes {sorted(names)} are neither the ten final classes "
                     "nor car_parts/v2's 29")


def to_ten_classes(detections, categories: Dict[int, str], kind: str, nms_iou: float = 0.5) -> list:
    """Detections relabeled to the ten final ids (``final_categories`` of the ten-class map)."""
    if kind == "ten":
        final_id = {n: i for i, n in final_categories(CLASS_MAP_ROBOFLOW, FINAL_CLASSES).items()}
        return [dict(d, category_id=final_id[categories[d["category_id"]]]) for d in detections]
    return to_final_detections(detections, categories, MAP_V2_TO_10, source_is_final=False,
                               nms_iou=nms_iou)


# ------------------------------------------------------------------------------ metrics
def metrics(m: dict, cat_names: Dict[int, str], groups: Optional[dict] = None) -> dict:
    """AP, AP50, AP per class and AP per group from one :func:`accumulate` result.

    A group's AP is the mean AP of its classes that have ground truth (as
    COCO does for the global mean); ``nan`` if none has.
    """
    per_class = {cat_names[cid]: v["AP"] for cid, v in m["per_class"].items()}
    out = {"AP": m["AP"], "AP50": m["AP50"], "per_class": per_class, "groups": {}}
    for group, classes in (GROUPS if groups is None else groups).items():
        values = [per_class[c] for c in classes if c in per_class]
        out["groups"][group] = float(np.mean(values)) if values else float("nan")
    return out


def bootstrap(evals: Dict[str, object], cat_names: Dict[int, str], n: int, seed: int,
              groups: Optional[dict] = None) -> Dict[str, dict]:
    """Paired bootstrap by image: the same ``n`` resamples for every entry of ``evals``.

    Returns, per entry, arrays of ``n`` values for AP, AP50, each class and
    each group (``nan`` where the resample has no ground truth for it).
    """
    img_ids = next(iter(evals.values())).img_ids
    for name, pe in evals.items():
        if pe.img_ids != img_ids:
            raise ValueError(f"{name} was evaluated on different images; the bootstrap must be paired")
    groups = GROUPS if groups is None else groups
    rng = np.random.default_rng(seed)
    out = {name: {"AP": np.full(n, np.nan), "AP50": np.full(n, np.nan),
                  "per_class": {c: np.full(n, np.nan) for c in cat_names.values()},
                  "groups": {g: np.full(n, np.nan) for g in groups}} for name in evals}
    for b in range(n):
        weights = np.bincount(rng.integers(0, len(img_ids), len(img_ids)), minlength=len(img_ids))
        for name, pe in evals.items():
            m = metrics(accumulate(pe, weights), cat_names, groups)
            out[name]["AP"][b], out[name]["AP50"][b] = m["AP"], m["AP50"]
            for c, v in m["per_class"].items():
                out[name]["per_class"][c][b] = v
            for g, v in m["groups"].items():
                out[name]["groups"][g][b] = v
    return out


def verdict(ci: dict) -> str:
    if ci["low"] is None:
        return "sin datos"
    if ci["low"] > 0:
        return "mejora"
    if ci["high"] < 0:
        return "empeora"
    return "no distinguible"


def _diff(a, b) -> np.ndarray:
    return np.asarray(b, dtype=float) - np.asarray(a, dtype=float)


def _value_ci(point: float, samples) -> dict:
    return {"value": None if point is None or np.isnan(point) else float(point),
            "ci95": interval(samples)}


def _diff_ci(point_a, point_b, samples_a, samples_b) -> dict:
    value = None
    if point_a is not None and point_b is not None and not (np.isnan(point_a) or np.isnan(point_b)):
        value = float(point_b - point_a)
    ci = interval(_diff(samples_a, samples_b))
    return {"value": value, "ci95": ci, "verdict": verdict(ci)}


def build_report(evals: Dict[str, object], gt: dict, cat_names: Dict[int, str], reference: str,
                 n_boot: int, seed: int, split: str, dataset: str, meta: Optional[dict] = None,
                 groups: Optional[dict] = None, warning: Optional[str] = None) -> dict:
    """Assemble the comparison from per-image evaluations.

    Args:
        evals: ``"<model>/bbox"`` and ``"<model>/segm"`` -> per-image eval, all
            on the same images and the ten class ids.
        gt: Ten-class ground truth dict, for instance counts.
        reference: Model name the others are compared against.
        cat_names: Only the classes to score (the evals must use these ids).
        groups: Class groups to average (default :data:`GROUPS`).
        warning: Extra warning shown first (e.g. :data:`EXTERNAL_WARNING`).
    """
    groups = GROUPS if groups is None else groups
    point = {k: metrics(accumulate(pe), cat_names, groups) for k, pe in evals.items()}
    boot = bootstrap(evals, cat_names, n_boot, seed, groups)
    img_ids = next(iter(evals.values())).img_ids
    cat_ids = next(iter(evals.values())).cat_ids
    counts = {cat_names[c]: n for c, n in instance_counts(gt, img_ids, cat_ids).items()}
    models = list(dict.fromkeys(k.split("/")[0] for k in evals))
    if reference not in models:
        raise ValueError(f"reference {reference!r} is not one of the models {models}")

    def summary(key):
        p, s = point[key], boot[key]
        return {"AP": _value_ci(p["AP"], s["AP"]), "AP50": _value_ci(p["AP50"], s["AP50"]),
                "groups": {g: _value_ci(p["groups"][g], s["groups"][g]) for g in groups},
                "per_class": {c: _value_ci(p["per_class"].get(c, float("nan")), s["per_class"][c])
                              for c in cat_names.values()}}

    def difference(key_ref, key):
        pr, pm, sr, sm = point[key_ref], point[key], boot[key_ref], boot[key]
        return {"AP": _diff_ci(pr["AP"], pm["AP"], sr["AP"], sm["AP"]),
                "AP50": _diff_ci(pr["AP50"], pm["AP50"], sr["AP50"], sm["AP50"]),
                "groups": {g: _diff_ci(pr["groups"][g], pm["groups"][g], sr["groups"][g], sm["groups"][g])
                           for g in groups},
                "per_class": {c: _diff_ci(pr["per_class"].get(c, float("nan")),
                                          pm["per_class"].get(c, float("nan")),
                                          sr["per_class"][c], sm["per_class"][c])
                              for c in cat_names.values()}}

    report = {
        "split": split, "dataset": dataset,
        "warning": " ".join(w for w in (warning, VAL_WARNING if split != "test" else None) if w) or None,
        "images": len(img_ids),
        "bootstrap": {"resamples": n_boot, "seed": seed, "paired_by": "image", "ci": "95% percentile"},
        "unreliable_below_instances": UNRELIABLE_BELOW,
        "groups": {g: list(c) for g, c in groups.items()},
        "instances": {c: {"count": counts[c], "unreliable": counts[c] < UNRELIABLE_BELOW}
                      for c in cat_names.values()},
        "reference": reference,
        "models": {},
        "differences_vs_reference": {},
    }
    for model in models:
        report["models"][model] = {iou: summary(f"{model}/{iou}") for iou in ("bbox", "segm")
                                   if f"{model}/{iou}" in evals}
        if model != reference:
            report["differences_vs_reference"][model] = {
                iou: difference(f"{reference}/{iou}", f"{model}/{iou}") for iou in ("bbox", "segm")
                if f"{model}/{iou}" in evals and f"{reference}/{iou}" in evals}
    if meta:
        report["meta"] = meta
    return report


# ------------------------------------------------------------------------------ Markdown
def _f(x, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def _vci(e, signed=False):
    ci = e["ci95"]
    rng = "-" if ci["low"] is None else f"[{_f(ci['low'], signed)}, {_f(ci['high'], signed)}]"
    return f"{_f(e['value'], signed)} {rng}"


IOU_NAMES = {"bbox": "caja", "segm": "máscara"}


def render_markdown(report: dict) -> str:
    lines = [f"# car_parts en 10 clases — `{report['dataset']}` split `{report['split']}` "
             f"({report['images']} imágenes)", ""]
    if report["warning"]:
        lines += [f"> **{report['warning']}**", ""]
    bs = report["bootstrap"]
    lines += [f"Bootstrap pareado por imagen: {bs['resamples']} remuestreos, semilla {bs['seed']}, "
              f"IC 95 % percentil. Valor [IC]. Referencia: **{report['reference']}**. "
              f"Clases con < {report['unreliable_below_instances']} instancias: *poco fiable*. "
              "Veredicto: mejora/empeora solo si el IC de la diferencia excluye cero.", "",
              "Grupos: " + "; ".join(f"`{g}` = {', '.join(c)}" for g, c in report["groups"].items()), ""]

    lines += ["## Resumen", "", "| modelo | IoU | AP | AP50 | " + " | ".join(report["groups"]) + " |",
              "|---|---|---|---|" + "---|" * len(report["groups"])]
    for model, by_iou in report["models"].items():
        for iou, s in by_iou.items():
            lines.append(f"| {model} | {IOU_NAMES[iou]} | {_vci(s['AP'])} | {_vci(s['AP50'])} | "
                         + " | ".join(_vci(s["groups"][g]) for g in report["groups"]) + " |")
    lines.append("")

    for model, by_iou in report["differences_vs_reference"].items():
        lines += [f"## {model} − {report['reference']}", "",
                  "| IoU | métrica | diferencia [IC] | veredicto |", "|---|---|---|---|"]
        for iou, d in by_iou.items():
            for label, e in [("AP", d["AP"]), ("AP50", d["AP50"])] + [
                    (f"grupo {g}", d["groups"][g]) for g in report["groups"]]:
                lines.append(f"| {IOU_NAMES[iou]} | {label} | {_vci(e, True)} | {e['verdict']} |")
        lines.append("")

    models = list(report["models"])
    for iou in ("bbox", "segm"):
        present = [m for m in models if iou in report["models"][m]]
        if not present:
            continue
        diffs = [m for m in present if iou in report["differences_vs_reference"].get(m, {})]
        headers = ["clase", "instancias"] + present + [f"{m} − {report['reference']}" for m in diffs] + ["nota"]
        lines += [f"## AP por clase ({IOU_NAMES[iou]}, AP@0.5:0.95)", "",
                  "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        for c, inst in report["instances"].items():
            cells = [_vci(report["models"][m][iou]["per_class"][c]) for m in present]
            cells += [_vci(report["differences_vs_reference"][m][iou]["per_class"][c], True) for m in diffs]
            row = [c, str(inst["count"])] + cells + ["poco fiable" if inst["unreliable"] else ""]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    hist = report.get("historical_v2_29_classes")
    if hist:
        lines += ["## Dato histórico: v2 en sus 29 clases (no comparable con lo anterior)", "",
                  "| modelo | IoU | AP | AP50 |", "|---|---|---|---|"]
        for model, by_iou in hist.items():
            for iou, s in by_iou.items():
                lines.append(f"| {model} | {IOU_NAMES[iou]} | {_vci(s['AP'])} | {_vci(s['AP50'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------- CLI
def parse_model(spec: str):
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"--model expects NAME=PATH, got {spec!r}")
    return name, path


def build_arg_parser():
    p = argparse.ArgumentParser(description="Compare car_parts models on the ten final classes.")
    p.add_argument("--model", action="append", type=parse_model, required=True,
                   help="NAME=CHECKPOINT; repeat for each model (ten-class or car_parts/v2).")
    p.add_argument("--reference", default=None,
                   help="NAME of the model the others are compared with (default: the first --model).")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="roboflow")
    p.add_argument("--split", default="test",
                   help="'test' for the verdict (default); the validation split only for selection.")
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nms-iou", type=float, default=0.5,
                   help="IoU for NMS within each final class after mapping v2's predictions.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def run(args) -> dict:
    names = [n for n, _ in args.model]
    if len(set(names)) != len(names):
        raise SystemExit(f"repeated model names: {names}")
    if args.reference is None:
        args.reference = names[0]
    if args.reference not in names:
        raise SystemExit(f"--reference {args.reference!r} is not one of {names}")
    cfg = DATASETS[args.dataset]
    if args.split not in (cfg.test_split, cfg.val_split, cfg.train_split):
        raise SystemExit(f"unknown split {args.split!r} for {args.dataset}")
    if args.split != cfg.test_split:
        print(VAL_WARNING)
    device = torch.device(args.device)

    ds = build_dataset(cfg, args.split)
    img_ids = sorted(ds.coco_image_ids)
    external = args.dataset == "carparts_seg"
    groups = EXTERNAL_GROUPS if external else GROUPS
    warning = EXTERNAL_WARNING if external else HITL_WARNING if args.dataset in ("hitl", "mix") else None
    cat_names = {i: n for i, n in ds.categories.items() if not external or n in EXTERNAL_CLASSES}
    cat_ids = list(cat_names)
    if warning:
        print(warning)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    hist_ds = None
    if args.dataset == "roboflow":
        hist_ds = build_dataset(MASK_CONFIG, args.split)
        if sorted(hist_ds.coco_image_ids) != img_ids:
            hist_ds = None  # not the same images: skip the historical row rather than mislead

    evals, hist_evals, meta = {}, {}, {"models": {}, "nms_iou": args.nms_iou}
    for name, path in args.model:
        version = load_version(path, device)
        kind = model_kind(version["categories"], version["class_map"])
        dets = collect_detections(version["model"], loader, device, with_masks=version["with_masks"])
        ten = to_ten_classes(dets, version["categories"], kind, args.nms_iou)
        evals[f"{name}/bbox"] = per_image_eval(ds.coco_gt, ten, "bbox", cat_ids, img_ids)
        if version["with_masks"]:
            evals[f"{name}/segm"] = per_image_eval(ds.coco_gt, ten, "segm", cat_ids, img_ids)
        if kind == "v2" and hist_ds is not None:
            v2_ids = list(hist_ds.categories)
            hist_evals[name] = {"bbox": per_image_eval(hist_ds.coco_gt, dets, "bbox", v2_ids, img_ids)}
            if version["with_masks"]:
                hist_evals[name]["segm"] = per_image_eval(hist_ds.coco_gt, dets, "segm", v2_ids, img_ids)
        meta["models"][name] = {"path": path, "kind": kind, "arch": version["arch"],
                                "epoch": version["epoch"], "with_masks": version["with_masks"],
                                "mapped_from_v2": kind == "v2"}
        del version
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = build_report(evals, ds.coco_gt, cat_names, args.reference, args.bootstrap, args.seed,
                          args.split, args.dataset, meta, groups=groups,
                          warning=warning)
    if hist_evals:
        hist_names = dict(hist_ds.categories)
        flat = {f"{m}/{iou}": pe for m, by in hist_evals.items() for iou, pe in by.items()}
        hb = {k: {"AP": v["AP"], "AP50": v["AP50"]} for k, v in
              bootstrap(flat, hist_names, args.bootstrap, args.seed, groups={}).items()}
        report["historical_v2_29_classes"] = {
            m: {iou: {"AP": _value_ci(accumulate(pe)["AP"], hb[f"{m}/{iou}"]["AP"]),
                      "AP50": _value_ci(accumulate(pe)["AP50"], hb[f"{m}/{iou}"]["AP50"])}
                for iou, pe in by.items()} for m, by in hist_evals.items()}

    os.makedirs(args.output, exist_ok=True)
    stem = os.path.join(args.output, f"compare_10c_{args.dataset}_{args.split}")
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    print(f"Wrote {stem}.json and {stem}.md")
    return report


def main(argv: Optional[List[str]] = None):
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
