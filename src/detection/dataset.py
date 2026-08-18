"""COCO-annotated object detection dataset for the "Car parts coco-segmentation" data.

Wraps the Roboflow-exported COCO JSON at
``data/raw/Car parts coco-segmentation/{split}/_annotations.coco.json`` as a
``torch.utils.data.Dataset`` compatible with ``torchvision.models.detection``
models (Faster R-CNN, RetinaNet, etc.), which expect bounding boxes only —
the polygon segmentation field in the source annotations is ignored here.

Category id 0 ("car-parts", the Roboflow root/supercategory) never appears on
any annotation in this dataset (verified against all three splits), so the
47 real part categories keep their original ids 1-47 and double as the
model's foreground class ids, with 0 reserved for "background" as
``torchvision`` detection models expect.
"""

import json
import os
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class CarPartsDetectionDataset(Dataset):
    """One split (train/valid/test) of the Car parts coco-segmentation dataset.

    Args:
        images_dir: Folder containing the split's ``.jpg`` images.
        ann_json_path: Path to that split's ``_annotations.coco.json``.
        skip_empty: If True (default), images with no bounding-box
            annotations are dropped rather than yielded with an empty
            target, since Faster R-CNN training is more stable when every
            batch element has at least one positive example.
    """

    def __init__(self, images_dir: str, ann_json_path: str, skip_empty: bool = True):
        self.images_dir = images_dir

        with open(ann_json_path, "r") as f:
            coco = json.load(f)

        self.categories = {c["id"]: c["name"] for c in coco["categories"] if c["id"] != 0}
        self.num_classes = max(self.categories) + 1  # +1 for background class 0

        anns_by_image_id = defaultdict(list)
        for ann in coco["annotations"]:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            anns_by_image_id[ann["image_id"]].append(ann)

        self.samples = []
        for img in coco["images"]:
            anns = anns_by_image_id.get(img["id"], [])
            if skip_empty and not anns:
                continue
            self.samples.append((img["file_name"], anns))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, anns = self.samples[idx]
        image = Image.open(os.path.join(self.images_dir, file_name)).convert("RGB")
        image = F.to_tensor(image)

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": idx,
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    """Batches (image, target) pairs of possibly-different object counts."""
    return tuple(zip(*batch))
