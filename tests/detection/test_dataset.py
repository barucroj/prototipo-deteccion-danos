import glob
import json
import os

import pytest
import torch

from src.detection.dataset import CarPartsDetectionDataset, collate_fn

DATA_ROOT = os.path.join("data", "raw", "Car parts coco-segmentation")
TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train")
TRAIN_ANN_PATH = os.path.join(TRAIN_IMAGES_DIR, "_annotations.coco.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(TRAIN_ANN_PATH),
    reason="Car parts coco-segmentation dataset not present under data/raw/",
)


@pytest.fixture(scope="module")
def dataset():
    return CarPartsDetectionDataset(TRAIN_IMAGES_DIR, TRAIN_ANN_PATH)


def test_category_ids_are_contiguous_and_exclude_background(dataset):
    with open(TRAIN_ANN_PATH) as f:
        coco = json.load(f)

    category_ids = {c["id"] for c in coco["categories"] if c["id"] != 0}
    assert set(dataset.categories) == category_ids
    assert dataset.num_classes == max(category_ids) + 1


def test_skips_images_without_annotations():
    with open(TRAIN_ANN_PATH) as f:
        coco = json.load(f)
    annotated_image_ids = {a["image_id"] for a in coco["annotations"] if a["bbox"][2] > 0 and a["bbox"][3] > 0}

    dataset = CarPartsDetectionDataset(TRAIN_IMAGES_DIR, TRAIN_ANN_PATH, skip_empty=True)
    assert len(dataset) == len(annotated_image_ids)


def test_getitem_returns_valid_target(dataset):
    image, target = dataset[0]

    assert image.dtype == torch.float32
    assert image.ndim == 3 and image.shape[0] == 3

    num_boxes = target["boxes"].shape[0]
    assert num_boxes > 0
    assert target["boxes"].shape == (num_boxes, 4)
    assert target["labels"].shape == (num_boxes,)

    # boxes must be well-formed (x2 > x1, y2 > y1) for torchvision detection models
    x1, y1, x2, y2 = target["boxes"].unbind(1)
    assert torch.all(x2 > x1)
    assert torch.all(y2 > y1)

    assert torch.all(target["labels"] >= 1)
    assert torch.all(target["labels"] < dataset.num_classes)


def test_collate_fn_batches_variable_length_targets(dataset):
    batch = [dataset[0], dataset[1]]
    images, targets = collate_fn(batch)

    assert len(images) == 2
    assert len(targets) == 2
    assert isinstance(targets[0], dict)
