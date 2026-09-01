"""Dataset-agnostic COCO-style detection dataset.

Wraps any COCO ``instances`` JSON as a ``torch.utils.data.Dataset`` yielding
``(image_tensor, target_dict)`` in the format ``torchvision.models.detection``
expects. Used for both detectors in this project (see
:mod:`src.detection.car_parts.config` and :mod:`src.detection.damage.config`),
which differ only in folder layout and which category ids are real classes.

Category ids are used **as-is** as model class ids, with 0 reserved for
background per the torchvision convention, so ``num_classes = max(id) + 1``.
That works because both datasets happen to have contiguous ids starting at 1
(car-parts 1-47 after dropping the Roboflow root category 0; CarDD 1-6). If a
future dataset has sparse or 0-based ids this needs a remapping table.

Instance masks are rasterized from the source polygons with ``cv2.fillPoly``,
which is exact for polygon segmentations and keeps this module free of a
``pycocotools`` dependency. RLE segmentations are not supported and raise.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Iterable

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
    """

    def __init__(
        self,
        images_dir: str,
        ann_json_path: str,
        skip_empty: bool = True,
        with_masks: bool = False,
        exclude_category_ids: Iterable[int] = (),
        transforms=None,
    ):
        self.images_dir = images_dir
        self.ann_json_path = ann_json_path
        self.with_masks = with_masks
        self.transforms = transforms

        with open(ann_json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        excluded = set(exclude_category_ids)
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


def build_dataset(cfg, split: str, data_root: str = None, **kwargs) -> CocoDetectionDataset:
    """Construct the dataset for one split of a :class:`DetectorConfig`."""
    images_dir, ann_path = cfg.split_paths(split, data_root)
    return CocoDetectionDataset(
        images_dir,
        ann_path,
        with_masks=cfg.with_masks,
        exclude_category_ids=cfg.exclude_category_ids,
        **kwargs,
    )


def collate_fn(batch):
    """Batches (image, target) pairs of possibly-different object counts."""
    return tuple(zip(*batch))
