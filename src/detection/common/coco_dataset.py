"""Dataset-agnostic COCO-style detection dataset.

Wraps any COCO ``instances`` JSON as a ``torch.utils.data.Dataset`` yielding
``(image_tensor, target_dict)`` in the format ``torchvision.models.detection``
expects. Used for both detectors in this project (see
:mod:`src.detection.car_parts.config` and :mod:`src.detection.damage.config`),
which differ only in folder layout and which category ids are real classes.

Category ids are used **as-is** as model class ids, with 0 reserved for
background per the torchvision convention, so ``num_classes = max(id) + 1``.
That works because both datasets happen to have contiguous ids starting at 1
(car-parts 1-47 after dropping the Roboflow root category 0; CarDD 1-6).

Optionally a ``class_map`` (original name -> final name, or ``None`` to
discard) rewrites the categories on load: see :func:`apply_class_map`. Final
ids are the final names sorted alphabetically -> 1..N, so they are contiguous
by construction.

Instance masks are rasterized from the source polygons with ``cv2.fillPoly``,
which is exact for polygon segmentations and keeps this module free of a
``pycocotools`` dependency. RLE segmentations are not supported and raise.
"""

from __future__ import annotations

import json
import os
import copy
from collections import Counter, defaultdict
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

try:  # cv2 is only needed when with_masks=True
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def polygons_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Rasterize one annotation's polygon segmentation to a uint8 {0,1} mask.

    Args:
        segmentation: COCO ``segmentation`` field: a list of polygons, each a
            flat ``[x0, y0, x1, y1, ...]`` list.
        height, width: Size of the image the polygons are in.

    Raises:
        TypeError: If given an RLE segmentation (a dict), which needs
            ``pycocotools`` to decode.
    """
    if isinstance(segmentation, dict):
        raise TypeError(
            "RLE segmentation is not supported without pycocotools; "
            "this dataset was expected to use polygon segmentations."
        )
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv-python is required for with_masks=True")

    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        if len(polygon) < 6:  # fewer than 3 points -> no area
            continue
        points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask


def final_categories(class_map: Mapping[str, Optional[str]]) -> Dict[int, str]:
    """``{final_id: final_name}``: the non-None final names, sorted, as ids 1..N."""
    names = sorted({name for name in class_map.values() if name is not None})
    return {i: name for i, name in enumerate(names, start=1)}


def apply_class_map(coco: dict, class_map: Mapping[str, Optional[str]],
                    exclude_category_ids: Iterable[int] = ()) -> Tuple[dict, dict]:
    """Rewrite a COCO dict's categories through ``class_map``.

    Categories in ``exclude_category_ids`` are removed first (as without a
    map). Every remaining category name must be a key of ``class_map`` and
    every key must be a category of the file, so a typo raises instead of
    silently dropping or keeping a class. Annotations of classes mapped to
    ``None`` are removed; merged classes share one final id.

    Returns:
        ``(new_coco, stats)``. ``new_coco`` is a copy with final
        ``categories`` and remapped ``annotations`` (images unchanged), usable
        as COCOeval ground truth. ``stats`` counts what the map changed:
        ``instances_discarded`` per original class, ``instances_per_final_class``,
        ``images_emptied`` (had annotations before the map, none after) and
        ``images_without_annotations`` (already empty in the source file).
    """
    excluded = set(exclude_category_ids)
    original = {c["id"]: c["name"] for c in coco["categories"] if c["id"] not in excluded}
    missing = sorted(set(original.values()) - set(class_map))
    unknown = sorted(set(class_map) - set(original.values()))
    if missing or unknown:
        raise ValueError(f"class_map does not match the dataset: missing {missing}, "
                         f"unknown {unknown}")

    final = final_categories(class_map)
    final_id = {name: i for i, name in final.items()}
    id_map = {cid: (final_id[class_map[name]] if class_map[name] is not None else None)
              for cid, name in original.items()}

    kept, discarded, per_final = [], Counter(), Counter()
    had_anns, has_anns = set(), set()
    for ann in coco["annotations"]:
        if ann["category_id"] in excluded:
            continue
        had_anns.add(ann["image_id"])
        new_id = id_map[ann["category_id"]]
        if new_id is None:
            discarded[original[ann["category_id"]]] += 1
            continue
        new_ann = dict(ann)
        new_ann["category_id"] = new_id
        kept.append(new_ann)
        per_final[final[new_id]] += 1
        has_anns.add(ann["image_id"])

    new_coco = {k: copy.deepcopy(v) for k, v in coco.items()
                if k not in ("categories", "annotations")}
    new_coco["categories"] = [{"id": i, "name": n, "supercategory": "none"}
                              for i, n in final.items()]
    new_coco["annotations"] = kept
    all_images = {img["id"] for img in coco["images"]}
    stats = {
        "final_categories": {str(i): n for i, n in final.items()},
        "instances_discarded": dict(sorted(discarded.items())),
        "instances_per_final_class": {n: per_final[n] for n in final.values()},
        "images_emptied": len(had_anns - has_anns),
        "images_without_annotations": len(all_images - had_anns),
    }
    return new_coco, stats


class CocoDetectionDataset(Dataset):
    """One split of a COCO-annotated detection dataset.

    Args:
        images_dir: Folder containing the split's image files.
        ann_json_path: Path to that split's COCO ``instances`` JSON.
        skip_empty: If True (default), images with no usable bounding-box
            annotations are dropped rather than yielded with an empty target,
            since Faster R-CNN training is more stable when every batch
            element has at least one positive example.
        with_masks: If True, also yield a ``masks`` tensor of shape
            ``(N, H, W)`` rasterized from the polygons, as Mask R-CNN needs.
        exclude_category_ids: Category ids to drop from ``categories`` (and
            from the class-count calculation) because they are not real
            foreground classes.
        transforms: Optional callable ``(image, target) -> (image, target)``
            applied after loading, for augmentation.
        class_map: Optional original-name -> final-name (or ``None``) table,
            applied on load via :func:`apply_class_map`. Images left without
            annotations by it are handled like empty images (dropped when
            ``skip_empty``); ``class_map_stats`` reports how many.

    Attributes:
        coco_gt: The (possibly remapped) COCO dict. Use it, not the JSON on
            disk, as COCOeval ground truth: with a class map the file's
            category ids no longer match the model's.
        class_map_stats: :func:`apply_class_map` stats, or ``None``.
    """

    def __init__(
        self,
        images_dir: str,
        ann_json_path: str,
        skip_empty: bool = True,
        with_masks: bool = False,
        exclude_category_ids: Iterable[int] = (),
        transforms=None,
        class_map: Optional[Mapping[str, Optional[str]]] = None,
    ):
        self.images_dir = images_dir
        self.ann_json_path = ann_json_path
        self.with_masks = with_masks
        self.transforms = transforms

        with open(ann_json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        excluded = set(exclude_category_ids)
        self.class_map = dict(class_map) if class_map is not None else None
        self.class_map_stats = None
        if class_map is not None:
            coco, self.class_map_stats = apply_class_map(coco, class_map, excluded)
            excluded = set()
        self.coco_gt = coco
        self.categories = {
            c["id"]: c["name"] for c in coco["categories"] if c["id"] not in excluded
        }
        if not self.categories:
            raise ValueError(f"{ann_json_path}: no categories left after exclusions")
        self.num_classes = max(self.categories) + 1  # +1 for background class 0

        anns_by_image_id = defaultdict(list)
        for ann in coco["annotations"]:
            if ann["category_id"] in excluded:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            anns_by_image_id[ann["image_id"]].append(ann)

        self.samples = []
        for img in coco["images"]:
            anns = anns_by_image_id.get(img["id"], [])
            if skip_empty and not anns:
                continue
            self.samples.append((img["id"], img["file_name"], img["width"], img["height"], anns))

    def __len__(self):
        return len(self.samples)

    @property
    def coco_image_ids(self):
        """The source COCO image ids of the samples that survived filtering.

        ``COCOeval`` is restricted to these so a capped or empty-filtered split
        is not scored against images that were never run through the model.
        """
        return [sample[0] for sample in self.samples]

    def __getitem__(self, idx):
        image_id, file_name, width, height, anns = self.samples[idx]
        image = Image.open(os.path.join(self.images_dir, file_name)).convert("RGB")
        image = F.to_tensor(image)

        boxes, labels, areas, iscrowd, masks = [], [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))
            if self.with_masks:
                masks.append(polygons_to_mask(ann["segmentation"], height, width))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            # The source COCO image id, not the position in this dataset, so
            # predictions can be scored against the original annotation file.
            "image_id": image_id,
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        if self.with_masks:
            stacked = np.stack(masks) if masks else np.zeros((0, height, width), np.uint8)
            target["masks"] = torch.as_tensor(stacked, dtype=torch.uint8)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def build_dataset(cfg, split: str, data_root: str = None, use_class_map: bool = True,
                  **kwargs) -> CocoDetectionDataset:
    """Construct the dataset for one split of a :class:`DetectorConfig`.

    Args:
        use_class_map: Apply ``cfg.class_map`` (if the config has one). False
            loads the original categories, e.g. to reproduce ``car_parts/v1``.
    """
    images_dir, ann_path = cfg.split_paths(split, data_root)
    return CocoDetectionDataset(
        images_dir,
        ann_path,
        with_masks=cfg.with_masks,
        exclude_category_ids=cfg.exclude_category_ids,
        class_map=cfg.class_map if use_class_map else None,
        **kwargs,
    )


def collate_fn(batch):
    """Batches (image, target) pairs of possibly-different object counts."""
    return tuple(zip(*batch))
