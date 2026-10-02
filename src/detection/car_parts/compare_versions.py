"""Compare car_parts v1 (47 original classes, boxes) against v2 (final classes, masks).

Usage:
    python -m src.detection.car_parts.compare_versions --v1 V1.pth --v2 V2.pth
        --split test --output DIR [--bootstrap 1000] [--seed 0] [--nms-iou 0.5]

Two experiments, on the same images of one split:

- **A — effect of reducing classes.** v1 scored on its 47 original classes
  (minus the ones the class map discards) against v1 with its predictions
  mapped through :data:`~src.detection.car_parts.class_map.CLASS_MAP` and
  scored on the final classes. Predictions that land in the same final class
  (e.g. ``left_front_door`` + ``right_front_door``) go through NMS within that
  class so one object is not counted twice.
- **B — effect of masks + augmentation (+ whatever else changed in v2).** v1
  mapped against v2, same images, same final classes. v2's mask AP is reported
  too (v1 has no masks, so there is no mask difference).

Metrics: COCO AP@0.5:0.95 and AP50 averaged over the experiment's classes only
(``params.catIds`` restricted), AP per class with the split's instance count,
and a paired bootstrap by image (same resampled images for every model) of
each model's AP and of the differences, with 95% percentile intervals. Classes
with fewer than 10 instances in the split are flagged unreliable.

Why a custom accumulate: ``COCOeval.evaluate`` deduplicates ``imgIds``, so a
bootstrap sample (images drawn with replacement) cannot be scored by handing
it to pycocotools. :func:`per_image_eval` runs pycocotools' per-image matching
once; :func:`accumulate` re-implements ``COCOeval.accumulate`` (area "all",
maxDets 100) with per-image weights. With all weights 1 it reproduces
pycocotools exactly (checked in the tests).

Rule: the best checkpoint of each version is chosen on ``valid``; the final
verdict is computed on ``test``. Running this on ``valid`` prints a warning
and records it in the report.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms

from src.detection.car_parts.class_map import CLASS_MAP, PENDING_DECISIONS
from src.detection.car_parts.config import CONFIG as V1_CONFIG
from src.detection.car_parts.config import MASK_CONFIG
from src.detection.common.coco_dataset import build_dataset, collate_fn, final_categories
from src.detection.common.coco_eval import collect_detections, load_coco_gt
from src.detection.common.model import build_model

UNRELIABLE_BELOW = 10
VAL_WARNING = ("WARNING: 'valid' is the split for choosing each version's best checkpoint. "
               "The final v1-vs-v2 verdict must be computed on 'test'.")


# --------------------------------------------------------------------------- evaluation
@dataclass
class PerImageEval:
    """pycocotools per-image matching results for one model, area 'all'."""
    cat_ids: List[int]
    img_ids: List[int]
    evals: List[List[Optional[dict]]]  # [category][image] -> COCOeval evalImg or None
    iou_thrs: np.ndarray
    rec_thrs: np.ndarray
    max_det: int


def _empty_results(coco_gt):
    from pycocotools.coco import COCO
    res = COCO()
    res.dataset = {"images": copy.deepcopy(coco_gt.dataset["images"]),
                   "categories": copy.deepcopy(coco_gt.dataset["categories"]),
                   "annotations": []}
    res.createIndex()
    return res


def per_image_eval(gt, detections, iou_type: str, cat_ids, img_ids) -> PerImageEval:
    """Run pycocotools' per-image evaluation once (no accumulate).

    Args:
        gt: COCO ground truth, as a dict or JSON path.
        detections: COCO-format results (``image_id``, ``category_id``,
            ``bbox`` xywh, ``score`` and, for ``segm``, ``segmentation``).
        cat_ids: Categories to score; AP is averaged over these only.
        img_ids: Images to score, in the order the bootstrap will index them.
    """
    from pycocotools.cocoeval import COCOeval

    coco_gt = load_coco_gt(gt)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = (coco_gt.loadRes(copy.deepcopy(list(detections))) if detections
                   else _empty_results(coco_gt))
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.params.catIds = list(cat_ids)
        ev.params.imgIds = list(img_ids)
        ev.evaluate()

    img_ids = list(ev.params.imgIds)  # pycocotools sorts and deduplicates
    n_img, n_area = len(img_ids), len(ev.params.areaRng)
    evals = [[ev.evalImgs[k * n_area * n_img + i] for i in range(n_img)]
             for k in range(len(ev.params.catIds))]
    return PerImageEval(list(ev.params.catIds), img_ids, evals,
                        ev.params.iouThrs, ev.params.recThrs, ev.params.maxDets[-1])


def accumulate(pe: PerImageEval, weights=None) -> dict:
    """AP@0.5:0.95, AP50 and per-class AP, with image ``i`` counted ``weights[i]`` times.

    Mirrors ``COCOeval.accumulate`` + ``summarize`` for area "all" and
    maxDets 100. A class with no (non-ignored) ground truth in the weighted
    sample is left out of the mean, as pycocotools does.

    Returns:
        ``{"AP", "AP50", "per_class": {cat_id: {"AP", "AP50"}}}``; AP/AP50 are
        ``nan`` when no class has ground truth.
    """
    n_img = len(pe.img_ids)
    w = np.ones(n_img, dtype=int) if weights is None else np.asarray(weights, dtype=int)
    order = np.repeat(np.arange(n_img), w)
    n_t, n_r = len(pe.iou_thrs), len(pe.rec_thrs)
    per_class = {}

    for k, cat_id in enumerate(pe.cat_ids):
        entries = [pe.evals[k][i] for i in order if pe.evals[k][i] is not None]
        if not entries:
            continue
        gt_ignore = np.concatenate([e["gtIgnore"] for e in entries])
        npig = int(np.count_nonzero(gt_ignore == 0))
        if npig == 0:
            continue
        scores = np.concatenate([e["dtScores"][:pe.max_det] for e in entries])
        precision = np.zeros((n_t, n_r))
        if scores.size:
            inds = np.argsort(-scores, kind="mergesort")
            dtm = np.concatenate([e["dtMatches"][:, :pe.max_det] for e in entries], axis=1)[:, inds]
            dt_ig = np.concatenate([e["dtIgnore"][:, :pe.max_det] for e in entries], axis=1)[:, inds]
            tps = np.logical_and(dtm, np.logical_not(dt_ig))
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dt_ig))
            tp_sum = np.cumsum(tps, axis=1).astype(float)
            fp_sum = np.cumsum(fps, axis=1).astype(float)
            recall = tp_sum / npig
            prec = tp_sum / (fp_sum + tp_sum + np.spacing(1))
            prec = np.maximum.accumulate(prec[:, ::-1], axis=1)[:, ::-1]  # envelope
            n_d = prec.shape[1]
            for t in range(n_t):
                idx = np.searchsorted(recall[t], pe.rec_thrs, side="left")
                ok = idx < n_d
                precision[t, ok] = prec[t, idx[ok]]
        per_class[cat_id] = {"AP": float(precision.mean()), "AP50": float(precision[0].mean())}

    if not per_class:
        return {"AP": float("nan"), "AP50": float("nan"), "per_class": {}}
    return {"AP": float(np.mean([v["AP"] for v in per_class.values()])),
            "AP50": float(np.mean([v["AP50"] for v in per_class.values()])),
            "per_class": per_class}


def paired_bootstrap(evals: Dict[str, PerImageEval], n: int, seed: int) -> Dict[str, dict]:
    """``n`` image resamples, identical for every model, of AP and AP50 per model."""
    img_ids = next(iter(evals.values())).img_ids
    for name, pe in evals.items():
        if pe.img_ids != img_ids:
            raise ValueError(f"{name} was evaluated on different images; the bootstrap must be paired")
    rng = np.random.default_rng(seed)
    out = {name: {"AP": np.empty(n), "AP50": np.empty(n)} for name in evals}
    for b in range(n):
        weights = np.bincount(rng.integers(0, len(img_ids), len(img_ids)), minlength=len(img_ids))
        for name, pe in evals.items():
            m = accumulate(pe, weights)
            out[name]["AP"][b] = m["AP"]
            out[name]["AP50"][b] = m["AP50"]
    return out


def interval(samples) -> dict:
    samples = np.asarray(samples, dtype=float)
    samples = samples[~np.isnan(samples)]
    if not samples.size:
        return {"low": None, "high": None}
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"low": float(low), "high": float(high)}


# ------------------------------------------------------------------------ class mapping
def to_final_detections(detections, categories: Dict[int, str], class_map, source_is_final: bool,
                        nms_iou: float = 0.5) -> list:
    """Re-label detections into the final classes of ``class_map``.

    Args:
        categories: The model's id -> name table (from its checkpoint).
        source_is_final: True when the model already predicts final classes
            (trained with this class map); then labels are only translated by
            name. False for a model on the original classes: names go through
            the map, discarded classes are dropped, and NMS at ``nms_iou``
            runs within each final class per image so merged classes (e.g.
            left and right door) do not double-count one object.
    """
    final_id = {name: i for i, name in final_categories(class_map).items()}
    out = []
    for det in detections:
        name = categories[det["category_id"]]
        target = name if source_is_final else class_map[name]
        if target is None:
            continue
        if target not in final_id:
            raise ValueError(f"model class {name!r} is not a final class of the class map")
        new = dict(det)
        new["category_id"] = final_id[target]
        out.append(new)
    if source_is_final or not out:
        return out

    by_image = {}
    for det in out:
        by_image.setdefault(det["image_id"], []).append(det)
    kept = []
    for dets in by_image.values():
        boxes = torch.tensor([[d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2],
                               d["bbox"][1] + d["bbox"][3]] for d in dets], dtype=torch.float32)
        scores = torch.tensor([d["score"] for d in dets], dtype=torch.float32)
        labels = torch.tensor([d["category_id"] for d in dets], dtype=torch.int64)
        keep = batched_nms(boxes, scores, labels, nms_iou)
        kept.extend(dets[i] for i in keep.tolist())
    return kept


def instance_counts(gt: dict, img_ids, cat_ids) -> Dict[int, int]:
    """Non-crowd ground-truth instances per category over ``img_ids``."""
    imgs, cats = set(img_ids), set(cat_ids)
    counts = {c: 0 for c in cat_ids}
    for a in gt["annotations"]:
        if a["image_id"] in imgs and a["category_id"] in cats and not a.get("iscrowd", 0):
            counts[a["category_id"]] += 1
    return counts


# ------------------------------------------------------------------------------ report
def _summary(point, boot) -> dict:
    return {"AP": point["AP"], "AP50": point["AP50"],
            "AP_ci95": interval(boot["AP"]), "AP50_ci95": interval(boot["AP50"])}


def _difference(point_a, point_b, boot_a, boot_b) -> dict:
    return {"AP": point_b["AP"] - point_a["AP"], "AP50": point_b["AP50"] - point_a["AP50"],
            "AP_ci95": interval(boot_b["AP"] - boot_a["AP"]),
            "AP50_ci95": interval(boot_b["AP50"] - boot_a["AP50"])}


def build_report(evals: Dict[str, PerImageEval], gts: Dict[str, dict], names: Dict[str, Dict[int, str]],
                 split: str, n_boot: int, seed: int, meta: Optional[dict] = None) -> dict:
    """Assemble both experiments from per-image evaluations.

    Args:
        evals: ``v1_original``, ``v1_mapped``, ``v2`` (box) and optionally
            ``v2_mask``, all on the same images.
        gts: ``{"original": gt dict, "final": gt dict}`` for instance counts.
        names: ``{"original": {id: name}, "final": {id: name}}``.
    """
    point = {name: accumulate(pe) for name, pe in evals.items()}
    boot = paired_bootstrap(evals, n_boot, seed)
    img_ids = evals["v1_mapped"].img_ids

    def class_rows(space, model_keys):
        cat_ids = evals[model_keys[0]].cat_ids
        counts = instance_counts(gts[space], img_ids, cat_ids)
        rows = []
        for cid in cat_ids:
            row = {"class": names[space][cid], "instances": counts[cid],
                   "unreliable": counts[cid] < UNRELIABLE_BELOW}
            for key in model_keys:
                row[key] = point[key]["per_class"].get(cid, {}).get("AP")
            rows.append(row)
        return rows

    final_models = ["v1_mapped", "v2"] + (["v2_mask"] if "v2_mask" in evals else [])
    report = {
        "split": split,
        "warning": VAL_WARNING if split != "test" else None,
        "images": len(img_ids),
        "bootstrap": {"resamples": n_boot, "seed": seed, "paired_by": "image", "ci": "95% percentile"},
        "unreliable_below_instances": UNRELIABLE_BELOW,
        "class_map_pending_decisions": sorted(PENDING_DECISIONS),
        "experiment_A_class_reduction": {
            "v1_original": _summary(point["v1_original"], boot["v1_original"]),
            "v1_mapped": _summary(point["v1_mapped"], boot["v1_mapped"]),
            "difference_mapped_minus_original": _difference(
                point["v1_original"], point["v1_mapped"], boot["v1_original"], boot["v1_mapped"]),
            "per_class_original": class_rows("original", ["v1_original"]),
        },
        "experiment_B_v2_vs_v1": {
            "v1_mapped": _summary(point["v1_mapped"], boot["v1_mapped"]),
            "v2_box": _summary(point["v2"], boot["v2"]),
            "difference_v2_minus_v1": _difference(
                point["v1_mapped"], point["v2"], boot["v1_mapped"], boot["v2"]),
            "v2_mask": _summary(point["v2_mask"], boot["v2_mask"]) if "v2_mask" in evals else None,
            "per_class_final": class_rows("final", final_models),
        },
    }
    if meta:
        report["meta"] = meta
    return report


def _f(x, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def _ci(c, signed=False):
    return "-" if c["low"] is None else f"[{_f(c['low'], signed)}, {_f(c['high'], signed)}]"


def render_markdown(report: dict) -> str:
    a, b = report["experiment_A_class_reduction"], report["experiment_B_v2_vs_v1"]
    lines = [f"# car_parts: v1 vs v2 — split `{report['split']}` ({report['images']} images)", ""]
    if report["warning"]:
        lines += [f"> **{report['warning']}**", ""]
    bs = report["bootstrap"]
    lines += [f"Paired bootstrap by image: {bs['resamples']} resamples, seed {bs['seed']}, "
              f"95% percentile intervals. Classes with < {report['unreliable_below_instances']} "
              f"instances in the split are marked *poco fiable*.", ""]
    if report["class_map_pending_decisions"]:
        lines += ["Class map still has pending decisions: "
                  + ", ".join(report["class_map_pending_decisions"]), ""]

    def summary_rows(title, entries):
        out = [f"## {title}", "", "| | AP@0.5:0.95 | IC 95% | AP50 | IC 95% |", "|---|---|---|---|---|"]
        for label, s, signed in entries:
            if s is None:
                continue
            out.append(f"| {label} | {_f(s['AP'], signed)} | {_ci(s['AP_ci95'], signed)} | "
                       f"{_f(s['AP50'], signed)} | {_ci(s['AP50_ci95'], signed)} |")
        return out + [""]

    lines += summary_rows("Experimento A — reducir clases (v1)", [
        ("v1, 47 clases originales", a["v1_original"], False),
        ("v1 mapeado a clases finales", a["v1_mapped"], False),
        ("diferencia (mapeado − original)", a["difference_mapped_minus_original"], True)])
    lines += summary_rows("Experimento B — v2 contra v1 (mismas clases finales)", [
        ("v1 mapeado (caja)", b["v1_mapped"], False),
        ("v2 (caja)", b["v2_box"], False),
        ("diferencia (v2 − v1, caja)", b["difference_v2_minus_v1"], True),
        ("v2 (máscara)", b["v2_mask"], False)])

    has_mask = b["v2_mask"] is not None
    lines += ["## AP por clase final (AP@0.5:0.95)", "",
              "| clase | instancias | v1 mapeado | v2 caja | " + ("v2 máscara | " if has_mask else "")
              + "v2 − v1 | nota |",
              "|---|---|---|---|" + ("---|" if has_mask else "") + "---|---|"]
    for r in b["per_class_final"]:
        diff = (r["v2"] - r["v1_mapped"]) if r["v2"] is not None and r["v1_mapped"] is not None else None
        lines.append(f"| {r['class']} | {r['instances']} | {_f(r['v1_mapped'])} | {_f(r['v2'])} | "
                     + (f"{_f(r.get('v2_mask'))} | " if has_mask else "")
                     + f"{_f(diff, True)} | {'poco fiable' if r['unreliable'] else ''} |")
    lines += ["", "## AP por clase original (v1, AP@0.5:0.95)", "",
              "| clase | instancias | v1 | nota |", "|---|---|---|---|"]
    for r in a["per_class_original"]:
        lines.append(f"| {r['class']} | {r['instances']} | {_f(r['v1_original'])} | "
                     f"{'poco fiable' if r['unreliable'] else ''} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------- CLI
def load_version(path: str, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "faster_rcnn")
    model = build_model(ckpt["num_classes"], arch=arch, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return {"model": model, "categories": {int(k): v for k, v in ckpt["categories"].items()},
            "class_map": ckpt.get("class_map"), "with_masks": bool(ckpt.get("with_masks", False)),
            "arch": arch, "epoch": ckpt.get("epoch"), "path": path}


def build_arg_parser():
    p = argparse.ArgumentParser(description="Compare car_parts v1 and v2 on one split.")
    p.add_argument("--v1", required=True, help="v1 checkpoint (original 47 classes).")
    p.add_argument("--v2", required=True, help="v2 checkpoint (final classes; a model on the "
                                                "original classes is mapped like v1).")
    p.add_argument("--split", choices=("valid", "test"), default="test",
                   help="'test' for the verdict (default); 'valid' only for model selection.")
    p.add_argument("--output", required=True, help="Folder for the JSON and Markdown report.")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nms-iou", type=float, default=0.5,
                   help="IoU for NMS within each final class after mapping v1's predictions.")
    p.add_argument("--data-root", default=MASK_CONFIG.data_root)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def run(args) -> dict:
    if args.split != "test":
        print(VAL_WARNING)
    device = torch.device(args.device)
    final = final_categories(CLASS_MAP)

    final_ds = build_dataset(MASK_CONFIG, args.split, args.data_root)
    orig_ds = build_dataset(V1_CONFIG, args.split, args.data_root)
    img_ids = sorted(set(final_ds.coco_image_ids) & set(orig_ds.coco_image_ids))
    keep = set(img_ids)
    orig_ds.samples = [s for s in orig_ds.samples if s[0] in keep]
    loader = DataLoader(orig_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    detections, versions = {}, {}
    for key in ("v1", "v2"):
        version = load_version(getattr(args, key), device)
        if key == "v1" and version["class_map"] is not None:
            raise SystemExit("--v1 must be a model on the original 47 classes (no class_map)")
        if key == "v2" and version["class_map"] not in (None, CLASS_MAP):
            raise SystemExit("--v2 was trained with a different class_map than the current "
                             "CLASS_MAP; its final classes would not line up")
        with_masks = key == "v2" and version["with_masks"]
        detections[key] = collect_detections(version["model"], loader, device, with_masks=with_masks)
        versions[key] = {k: v for k, v in version.items() if k != "model"}
        del version
        if device.type == "cuda":
            torch.cuda.empty_cache()

    v1_mapped = to_final_detections(detections["v1"], versions["v1"]["categories"], CLASS_MAP,
                                    source_is_final=False, nms_iou=args.nms_iou)
    v2_final = to_final_detections(detections["v2"], versions["v2"]["categories"], CLASS_MAP,
                                   source_is_final=versions["v2"]["class_map"] is not None,
                                   nms_iou=args.nms_iou)
    orig_cat_ids = [cid for cid, name in orig_ds.categories.items() if CLASS_MAP[name] is not None]
    final_cat_ids = list(final)

    evals = {
        "v1_original": per_image_eval(orig_ds.coco_gt, detections["v1"], "bbox", orig_cat_ids, img_ids),
        "v1_mapped": per_image_eval(final_ds.coco_gt, v1_mapped, "bbox", final_cat_ids, img_ids),
        "v2": per_image_eval(final_ds.coco_gt, v2_final, "bbox", final_cat_ids, img_ids),
    }
    if versions["v2"]["with_masks"]:
        evals["v2_mask"] = per_image_eval(final_ds.coco_gt, v2_final, "segm", final_cat_ids, img_ids)

    meta = {"v1": {k: versions["v1"][k] for k in ("path", "arch", "epoch")},
            "v2": {k: versions["v2"][k] for k in ("path", "arch", "epoch", "with_masks")},
            "v2_mapped_from_original_classes": versions["v2"]["class_map"] is None,
            "nms_iou": args.nms_iou}
    report = build_report(
        evals, {"original": orig_ds.coco_gt, "final": final_ds.coco_gt},
        {"original": dict(orig_ds.categories), "final": final},
        args.split, args.bootstrap, args.seed, meta)

    os.makedirs(args.output, exist_ok=True)
    stem = os.path.join(args.output, f"compare_{args.split}")
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    print(f"Wrote {stem}.json and {stem}.md")
    return report


def main(argv=None):
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
