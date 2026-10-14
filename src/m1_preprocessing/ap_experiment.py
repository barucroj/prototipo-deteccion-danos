"""Does M1 help the detectors? COCO AP with and without preprocessing.

Runs one trained detector over a split several times, each time with a
different M1 variant applied to the images before inference, and compares
every variant against the raw images:

* ``raw``      — no preprocessing (how every current model was trained);
* ``specular`` — highlight removal only (the M1 default);
* ``clahe``    — CLAHE only;
* ``both``     — highlight removal, then CLAHE.

The steps are photometric, so the ground truth is the same for all variants.
The comparison is paired by image: 1000 bootstrap resamples (the same for
every variant), 95 % percentile intervals of each AP and of each difference
against ``raw``. A difference whose interval contains 0 is reported as not
distinguishable from noise.

Scope: this is **inference only**. The models were trained on raw images, so
it answers "does M1 help or hurt the current models", not "would a model
trained with M1 be better" — that needs a retraining.

Usage::

    python -m src.m1_preprocessing.ap_experiment --detector damage
    python -m src.m1_preprocessing.ap_experiment --detector parts \\
        --checkpoint models/checkpoints/car_parts/v2/best_model.pth

Writes ``m1_effect_{split}.json`` and ``.md`` to ``--output`` (default: an
``m1_effect/`` folder next to the checkpoint).

Retraining mode. ``--retrained CKPT`` answers the second question: it takes a
model trained *with* M1 (``--preprocess`` of the trainer, recorded in the
checkpoint) and compares, on the same images,

* ``raw``                 — ``--checkpoint`` (trained on raw images) on raw images;
* ``retrained``           — the retrained model on images preprocessed the way
  it was trained;
* ``retrained_raw_input`` — the retrained model on raw images, to see how much
  it depends on the preprocessing.

::

    python -m src.m1_preprocessing.ap_experiment --detector parts \\
        --retrained models/checkpoints/car_parts/v3_m1_specular/best_model.pth

Writes ``m1_retrain_{split}.json`` and ``.md`` next to the retrained
checkpoint. Both models share recipe and seed, but training is not
bit-deterministic: a small difference can be run-to-run noise.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.detection_common.coco_dataset import build_dataset, collate_fn
from src.detection_common.coco_eval import collect_detections
from src.m1_preprocessing.torch_transform import (BASELINE, VARIANTS, PreprocessTransform,
                                                  build_preprocess_transform)
from src.m2_parts.compare_versions import (UNRELIABLE_BELOW, PerImageEval, accumulate,
                                           instance_counts, interval, load_version,
                                           paired_bootstrap, per_image_eval)
from src.m2_parts.config import MASK_CONFIG as PARTS_CONFIG
from src.m3_damage.detector.config import CONFIG as DAMAGE_CONFIG

VARIANT_LABELS = {
    BASELINE: "sin preprocesamiento",
    "specular": "reflejos",
    "clahe": "CLAHE",
    "both": "reflejos + CLAHE",
}

DETECTORS = {
    "parts": (PARTS_CONFIG, os.path.join("models", "checkpoints", "car_parts", "v2", "best_model.pth")),
    "damage": (DAMAGE_CONFIG, os.path.join("models", "checkpoints", "damage", "v2", "best_model.pth")),
}

IOU_LABELS = {"bbox": "caja", "segm": "máscara"}
VAL_WARNING = ("WARNING: this split chose the checkpoint; quote the M1 effect from 'test'.")


# ------------------------------------------------------------------------------ report
def _point(pe: PerImageEval) -> dict:
    return accumulate(pe)


def build_report(evals: Dict[str, Dict[str, PerImageEval]], gt: dict, categories: Dict[int, str],
                 n_bootstrap: int, seed: int, meta: dict) -> dict:
    """Compare every variant against ``raw``.

    Args:
        evals: ``{iou_type: {variant: PerImageEval}}``; every iou type must
            include the ``raw`` variant, and all variants the same images.
        gt: COCO ground-truth dict the evaluations were scored against.
        categories: ``{category_id: name}`` of the scored classes.
        n_bootstrap / seed: Paired bootstrap settings.
        meta: Free-form run description, copied into the report.

    Returns:
        ``{"meta", "metrics": {iou_type: {"variants", "differences", "per_class"}}}``.
        ``differences[variant]`` is ``variant - raw`` with its 95 % interval
        and ``distinguible`` (interval excludes 0).
    """
    metrics = {}
    for iou_type, by_variant in evals.items():
        if BASELINE not in by_variant:
            raise ValueError(f"{iou_type}: the '{BASELINE}' variant is required as the baseline")
        points = {name: _point(pe) for name, pe in by_variant.items()}
        boots = paired_bootstrap(by_variant, n_bootstrap, seed)
        base = by_variant[BASELINE]
        counts = instance_counts(gt, base.img_ids, base.cat_ids)

        variants = {
            name: {"AP": points[name]["AP"], "AP50": points[name]["AP50"],
                   "AP_ci95": interval(boots[name]["AP"]), "AP50_ci95": interval(boots[name]["AP50"])}
            for name in by_variant
        }
        differences = {}
        for name in by_variant:
            if name == BASELINE:
                continue
            entry = {}
            for stat in ("AP", "AP50"):
                ci = interval(boots[name][stat] - boots[BASELINE][stat])
                entry[stat] = points[name][stat] - points[BASELINE][stat]
                entry[f"{stat}_ci95"] = ci
                entry[f"{stat}_distinguible"] = (
                    ci["low"] is not None and (ci["low"] > 0 or ci["high"] < 0))
            differences[name] = entry

        per_class = []
        for cat_id in base.cat_ids:
            row = {"clase": categories.get(cat_id, str(cat_id)), "instancias": counts.get(cat_id, 0),
                   "poco_fiable": counts.get(cat_id, 0) < UNRELIABLE_BELOW}
            for name in by_variant:
                row[name] = points[name]["per_class"].get(cat_id, {}).get("AP")
            per_class.append(row)

        metrics[iou_type] = {"variants": variants, "differences": differences, "per_class": per_class}
    return {"meta": meta, "metrics": metrics}


def _f(value, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _ci(ci: dict, signed: bool = False) -> str:
    if ci["low"] is None:
        return "-"
    return f"[{_f(ci['low'], signed)}, {_f(ci['high'], signed)}]"


def render_markdown(report: dict) -> str:
    meta = report["meta"]
    labels = dict(VARIANT_LABELS, **meta.get("labels", {}))
    lines = [
        f"# Efecto de M1 en `{meta['detector']}` — split `{meta['split']}` ({meta['images']} imágenes)",
        "",
    ]
    if meta.get("retrained"):
        r = meta["retrained"]
        lines += [
            f"Modelo de referencia: `{meta['checkpoint']}` (época {meta['epoch']}), entrenado con "
            "imágenes crudas.",
            "",
            f"Modelo reentrenado: `{r['checkpoint']}` (época {r['epoch']}), entrenado con M1 "
            f"`{r['preprocess']}`. Misma receta y semilla; el entrenamiento no es determinista bit "
            "a bit, así que una diferencia pequeña puede ser ruido entre corridas.",
            "",
        ]
    else:
        lines += [
            f"Checkpoint: `{meta['checkpoint']}` (época {meta['epoch']}). Solo inferencia: el modelo "
            "se entrenó con imágenes crudas.",
            "",
        ]
    lines += [
        f"Bootstrap pareado por imagen: {meta['bootstrap']} remuestreos, semilla {meta['seed']}, "
        f"intervalos percentil 95 %. Clases con < {UNRELIABLE_BELOW} instancias marcadas *poco fiable*.",
        "",
    ]
    for iou_type, block in report["metrics"].items():
        label = IOU_LABELS.get(iou_type, iou_type)
        lines += [f"## AP de {label}", "",
                  "| variante | AP@0.5:0.95 | IC 95% | AP50 | IC 95% |", "|---|---|---|---|---|"]
        for name, v in block["variants"].items():
            lines.append(f"| {labels.get(name, name)} | {_f(v['AP'])} | {_ci(v['AP_ci95'])} "
                         f"| {_f(v['AP50'])} | {_ci(v['AP50_ci95'])} |")
        lines += ["", f"### Diferencia contra la referencia sin preprocesar ({label})", "",
                  "| variante | ΔAP | IC 95% | | ΔAP50 | IC 95% | |", "|---|---|---|---|---|---|---|"]
        for name, d in block["differences"].items():
            verdict = {True: "distinguible", False: "no distinguible"}
            lines.append(
                f"| {labels.get(name, name)} | {_f(d['AP'], True)} | {_ci(d['AP_ci95'], True)} "
                f"| {verdict[d['AP_distinguible']]} | {_f(d['AP50'], True)} "
                f"| {_ci(d['AP50_ci95'], True)} | {verdict[d['AP50_distinguible']]} |")
        names = list(block["variants"])
        lines += ["", f"### AP por clase ({label}, AP@0.5:0.95)", "",
                  "| clase | instancias | " + " | ".join(labels.get(n, n) for n in names) + " | nota |",
                  "|---|---|" + "---|" * (len(names) + 1)]
        for row in block["per_class"]:
            lines.append(f"| {row['clase']} | {row['instancias']} | "
                         + " | ".join(_f(row[n]) for n in names)
                         + f" | {'poco fiable' if row['poco_fiable'] else ''} |")
        lines.append("")

    timing = meta.get("preprocess_ms")
    if timing:
        lines += ["## Tiempo del preprocesamiento por imagen (CPU)", "",
                  "| variante | mediana (ms) | media (ms) | máx. (ms) |", "|---|---|---|---|"]
        for name, t in timing.items():
            lines.append(f"| {labels.get(name, name)} | {t['median']:.1f} | {t['mean']:.1f} "
                         f"| {t['max']:.1f} |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------- CLI
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="COCO AP of a detector with and without M1.")
    parser.add_argument("--detector", required=True, choices=sorted(DETECTORS))
    parser.add_argument("--checkpoint", default=None,
                        help="Model trained on raw images. Default: the current model of the "
                             "detector (car_parts/v2, damage/v2).")
    parser.add_argument("--retrained", default=None,
                        help="A model trained with M1 (--preprocess). Switches to retraining "
                             "mode: it is compared against --checkpoint instead of the variants.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS),
                        help=f"'{BASELINE}' is always included. Ignored with --retrained.")
    parser.add_argument("--output", default=None,
                        help="Default: <checkpoint folder>/m1_effect, or with --retrained "
                             "<retrained checkpoint folder>/m1_retrain.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-images", type=int, default=None, help="Cap the split (smoke tests).")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def checkpoint_preprocess(path: str) -> str:
    """The M1 variant a checkpoint was trained with (``raw`` if it predates the flag)."""
    return torch.load(path, map_location="cpu", weights_only=False).get("preprocess", BASELINE)


def plan_arms(checkpoint: str, retrained, variants, base_variant: str, trained_variant=None):
    """Which (name, checkpoint, input variant) evaluations to run, and their labels.

    Args:
        checkpoint: The model trained on raw images.
        retrained: The model trained with M1, or ``None`` for the variants mode.
        variants: Variant names for the variants mode.
        base_variant / trained_variant: What each checkpoint was trained with.

    Returns:
        ``(arms, labels)``; ``labels`` is empty in the variants mode.
    """
    if base_variant != BASELINE:
        raise SystemExit(f"--checkpoint was trained with M1 '{base_variant}'; it must be a model "
                         "trained on raw images (pass the M1 one as --retrained).")
    if not retrained:
        names = [BASELINE] + [v for v in variants if v != BASELINE]
        return [(name, checkpoint, name) for name in names], {}

    if trained_variant == BASELINE:
        raise SystemExit("--retrained was trained on raw images (no --preprocess); "
                         "there is nothing to compare.")
    m1 = VARIANT_LABELS[trained_variant]
    arms = [(BASELINE, checkpoint, BASELINE),
            ("retrained", retrained, trained_variant),
            ("retrained_raw_input", retrained, BASELINE)]
    labels = {BASELINE: "referencia (entrenado y evaluado sin M1)",
              "retrained": f"reentrenado con {m1}, entrada con {m1}",
              "retrained_raw_input": f"reentrenado con {m1}, entrada sin preprocesar"}
    return arms, labels


def _timing(transform: PreprocessTransform) -> dict:
    ms = np.asarray(transform.seconds) * 1000.0
    return {"median": float(np.median(ms)), "mean": float(ms.mean()), "max": float(ms.max())}


def run(args) -> dict:
    cfg, default_checkpoint = DETECTORS[args.detector]
    checkpoint = args.checkpoint or default_checkpoint
    if args.split != cfg.test_split:
        print(VAL_WARNING)
    device = torch.device(args.device)

    trained_variant = checkpoint_preprocess(args.retrained) if args.retrained else None
    arms, labels = plan_arms(checkpoint, args.retrained, args.variants,
                             checkpoint_preprocess(checkpoint), trained_variant)
    if args.retrained:
        output = args.output or os.path.join(os.path.dirname(args.retrained), "m1_retrain")
        stem_name = f"m1_retrain_{args.split}"
    else:
        output = args.output or os.path.join(os.path.dirname(checkpoint), "m1_effect")
        stem_name = f"m1_effect_{args.split}"

    versions, dataset, loader = {}, None, None
    evals: Dict[str, Dict[str, PerImageEval]] = {}
    timing = {}
    for name, path, variant in arms:
        if path not in versions:
            for old in versions.values():  # one model on the GPU at a time
                old["model"].to("cpu")
            versions[path] = load_version(path, device)
        version = versions[path]
        version["model"].to(device)

        if dataset is None:
            with_masks = version["with_masks"]
            dataset = build_dataset(replace(cfg, with_masks=with_masks), args.split, args.data_root,
                                    use_class_map=version["class_map"] is not None)
            if args.max_images is not None:
                dataset.samples = dataset.samples[:args.max_images]
            img_ids = dataset.coco_image_ids
            cat_ids = list(dataset.categories)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
            iou_types = ["bbox"] + (["segm"] if with_masks else [])
            evals = {iou: {} for iou in iou_types}
        if dict(dataset.categories) != version["categories"] or version["with_masks"] != with_masks:
            raise SystemExit(f"{path}: its classes or masks do not match the dataset's; "
                             "was it trained with a different class_map?")

        dataset.transforms = build_preprocess_transform(variant)
        print(f"[{name}] {labels.get(name, VARIANT_LABELS.get(name, name))}")
        detections = collect_detections(version["model"], loader, device, with_masks=with_masks)
        for iou in iou_types:
            evals[iou][name] = per_image_eval(dataset.coco_gt, detections, iou, cat_ids, img_ids)
        if dataset.transforms is not None:
            timing[name] = _timing(dataset.transforms)
        del detections
    dataset.transforms = None

    base = versions[checkpoint]
    meta = {
        "detector": cfg.name, "checkpoint": checkpoint, "epoch": base["epoch"],
        "arch": base["arch"], "split": args.split, "images": len(img_ids),
        "bootstrap": args.bootstrap, "seed": args.seed, "device": str(device),
        "inference_only": not args.retrained,
        "arms": [{"name": name, "checkpoint": path, "input_variant": variant,
                  "config": asdict(VARIANTS[variant]) if VARIANTS[variant] is not None else None}
                 for name, path, variant in arms],
        "preprocess_ms": timing,
    }
    if args.retrained:
        meta["labels"] = labels
        meta["retrained"] = {"checkpoint": args.retrained, "epoch": versions[args.retrained]["epoch"],
                             "preprocess": trained_variant}
    report = build_report(evals, dataset.coco_gt, dict(dataset.categories),
                          args.bootstrap, args.seed, meta)

    os.makedirs(output, exist_ok=True)
    stem = os.path.join(output, stem_name)
    with open(stem + ".json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(stem + ".md", "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    print(f"Wrote {stem}.json and {stem}.md")
    return report


def main(argv=None):
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
