"""Merge Roboflow Car-Parts and the Humans in the Loop part images into one COCO set.

Writes ``data/raw/parts_mix_coco/instances_{split}.json`` for
:data:`src.m2_parts.config.MASK_10_CONFIG` (``v3_10c``) and
:data:`~src.m2_parts.config.MASK_MIX_CONFIG` (``v3_mix``):

- ``train``          Roboflow train + HITL train
- ``train_roboflow`` Roboflow train only
- ``val``            Roboflow valid + HITL val
- ``test``           Roboflow test + HITL test

Categories keep the original names with the source as prefix
(``roboflow/back_bumper``, ``hitl/Back-bumper``) and the ten final classes come
from :data:`src.m2_parts.class_map_hitl.CLASS_MAP_MIX`; every file lists the
categories of both sources so the map always matches. Roboflow's root
category (id 0, never on an annotation) is dropped. ``file_name`` is relative
to ``data/raw``; each image keeps ``source`` and ``source_id`` (its id in the
original file). Image and annotation ids are renumbered from 1.

The image order is shuffled with a fixed seed: the trainer's
``--max-train-images N`` takes the **first** N images, so a short timing run
gets a sample of both sources instead of only Roboflow.

Usage:
    python -m src.m2_parts.mix_dataset [--output data/raw/parts_mix_coco] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.detection_common.coco_dataset import apply_class_map
from src.m2_parts.class_map_10 import FINAL_CLASSES
from src.m2_parts.class_map_hitl import CLASS_MAP_MIX, MIX_SEPARATOR
from src.m2_parts.config import HITL_10_CONFIG, ROBOFLOW_10_CONFIG

DATA_RAW = os.path.join("data", "raw")
DEFAULT_OUTPUT = os.path.join(DATA_RAW, "parts_mix_coco")

#: ``{mixed split: [(source, config, source split), ...]}``
MIX_SPLITS = {
    "train": [("roboflow", ROBOFLOW_10_CONFIG, "train"), ("hitl", HITL_10_CONFIG, "train")],
    "train_roboflow": [("roboflow", ROBOFLOW_10_CONFIG, "train")],
    "val": [("roboflow", ROBOFLOW_10_CONFIG, "valid"), ("hitl", HITL_10_CONFIG, "val")],
    "test": [("roboflow", ROBOFLOW_10_CONFIG, "test"), ("hitl", HITL_10_CONFIG, "test")],
}

#: (source, coco dict, images dir relative to data/raw, excluded category ids)
Part = Tuple[str, dict, str, Iterable[int]]


def merge_coco(parts: Sequence[Part], all_categories: Sequence[str], seed: Optional[int] = 0) -> dict:
    """One COCO dict from several, with ``<source>/<name>`` categories.

    Args:
        parts: ``(source, coco, images_dir, exclude_category_ids)``;
            ``images_dir`` is prefixed to each ``file_name`` (with ``/``).
        all_categories: Every prefixed category name, written to the file in
            this order (ids 1..N), even those absent from ``parts``.
        seed: Shuffles the image order (``None`` keeps it).
    """
    cat_id = {name: i for i, name in enumerate(all_categories, start=1)}
    images, per_image = [], {}
    for source, coco, images_dir, excluded in parts:
        excluded = set(excluded)
        names = {c["id"]: f"{source}{MIX_SEPARATOR}{c['name']}" for c in coco["categories"]
                 if c["id"] not in excluded}
        unknown = sorted(set(names.values()) - set(cat_id))
        if unknown:
            raise ValueError(f"{source}: categories {unknown} are not in all_categories")
        anns = {}
        for ann in coco["annotations"]:
            if ann["category_id"] in excluded:
                raise ValueError(f"{source}: annotation {ann['id']} uses excluded category "
                                 f"{ann['category_id']}")
            new = dict(ann, category_id=cat_id[names[ann["category_id"]]])
            anns.setdefault(ann["image_id"], []).append(new)
        for img in coco["images"]:
            key = (source, img["id"])
            if key in per_image:
                raise ValueError(f"{source}: image id {img['id']} repeated")
            file_name = f"{images_dir}/{img['file_name']}" if images_dir else img["file_name"]
            images.append(dict(img, file_name=file_name, source=source, source_id=img["id"]))
            per_image[key] = anns.get(img["id"], [])
    if seed is not None:
        random.Random(seed).shuffle(images)
    out_images, out_anns = [], []
    for new_id, img in enumerate(images, start=1):
        for ann in per_image[(img["source"], img["source_id"])]:
            out_anns.append(dict(ann, id=len(out_anns) + 1, image_id=new_id))
        out_images.append(dict(img, id=new_id))
    return {"images": out_images, "annotations": out_anns,
            "categories": [{"id": i, "name": n, "supercategory": "car_part"} for n, i in cat_id.items()]}


def _relative_images_dir(cfg, split: str) -> str:
    images_dir, _ = cfg.split_paths(split)
    return os.path.relpath(images_dir, DATA_RAW).replace(os.sep, "/")


def build_split(split: str, seed: Optional[int] = 0) -> dict:
    """The mixed COCO dict of one split of :data:`MIX_SPLITS`, read from the source files."""
    parts = []
    for source, cfg, source_split in MIX_SPLITS[split]:
        _, ann_path = cfg.split_paths(source_split)
        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)
        parts.append((source, coco, _relative_images_dir(cfg, source_split), cfg.exclude_category_ids))
    return merge_coco(parts, sorted(CLASS_MAP_MIX), seed)


def class_counts(coco: dict) -> Dict[str, dict]:
    """Images per source and instances per source and final class (after the map)."""
    mapped, _ = apply_class_map(coco, CLASS_MAP_MIX, final_classes=FINAL_CLASSES)
    names = {c["id"]: c["name"] for c in mapped["categories"]}
    source = {img["id"]: img["source"] for img in coco["images"]}
    per = {}
    for ann in mapped["annotations"]:
        per.setdefault(source[ann["image_id"]], Counter())[names[ann["category_id"]]] += 1
    return {"images": dict(sorted(Counter(source.values()).items())),
            "instances": {s: {c: per.get(s, Counter())[c] for c in sorted(FINAL_CLASSES)}
                          for s in sorted(set(source.values()))}}


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Merge Roboflow and HITL car parts into one COCO set.")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=0, help="Seed of the image order (see --max-train-images).")
    args = p.parse_args(argv)
    os.makedirs(args.output, exist_ok=True)
    summary = {}
    for split in MIX_SPLITS:
        coco = build_split(split, args.seed)
        with open(os.path.join(args.output, f"instances_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(coco, f)
        summary[split] = class_counts(coco)
    with open(os.path.join(args.output, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
