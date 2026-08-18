"""CLI entrypoint: train the car-parts Faster R-CNN detector.

Usage:
    python -m src.detection.train [--data-root DIR] [--output DIR]
        [--epochs N] [--batch-size N] [--lr LR] [--device cpu|cuda]

Trains on the ``train`` split, evaluates mAP@0.5 on ``valid`` after every
epoch, and writes ``best_model.pth`` (highest validation mAP so far),
``last_model.pth`` (most recent epoch), and ``metrics.json`` (per-epoch
loss/mAP history) to ``--output``.
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from src.detection.dataset import CarPartsDetectionDataset, collate_fn
from src.detection.engine import evaluate, train_one_epoch
from src.detection.model import build_model


def _split_paths(data_root: str, split: str):
    images_dir = os.path.join(data_root, split)
    ann_path = os.path.join(images_dir, "_annotations.coco.json")
    return images_dir, ann_path


def main():
    parser = argparse.ArgumentParser(description="Train the car-parts object detector.")
    parser.add_argument(
        "--data-root",
        default=os.path.join("data", "raw", "Car parts coco-segmentation"),
        help="Dataset root containing train/valid/test split folders.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("models", "checkpoints", "test"),
        help="Folder to write checkpoints and metrics.json to.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="If set, cap the training set to this many images (for quick smoke tests).",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device)

    train_images_dir, train_ann_path = _split_paths(args.data_root, "train")
    valid_images_dir, valid_ann_path = _split_paths(args.data_root, "valid")

    train_dataset = CarPartsDetectionDataset(train_images_dir, train_ann_path)
    valid_dataset = CarPartsDetectionDataset(valid_images_dir, valid_ann_path)

    if args.max_train_images is not None:
        train_dataset.samples = train_dataset.samples[: args.max_train_images]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    print(f"Train images: {len(train_dataset)}  Valid images: {len(valid_dataset)}  "
          f"Classes (incl. background): {train_dataset.num_classes}  Device: {device}")

    model = build_model(num_classes=train_dataset.num_classes, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    history = []
    best_map = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_map, _ = evaluate(model, valid_loader, device)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_mAP@0.5={val_map:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_map50": val_map})

        torch.save(
            {"model_state_dict": model.state_dict(), "num_classes": train_dataset.num_classes,
             "categories": train_dataset.categories, "epoch": epoch},
            os.path.join(args.output, "last_model.pth"),
        )
        if val_map > best_map:
            best_map = val_map
            torch.save(
                {"model_state_dict": model.state_dict(), "num_classes": train_dataset.num_classes,
                 "categories": train_dataset.categories, "epoch": epoch, "val_map50": val_map},
                os.path.join(args.output, "best_model.pth"),
            )

        with open(os.path.join(args.output, "metrics.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"Done. Best val mAP@0.5 = {best_map:.4f}. Checkpoints saved to {args.output}")


if __name__ == "__main__":
    main()
