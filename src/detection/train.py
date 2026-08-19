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
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    print(f"\n{'='*80}")
    print(f"Train images: {len(train_dataset)}  |  Valid images: {len(valid_dataset)}")
    print(f"Classes: {train_dataset.num_classes}  |  Device: {device}")
    print(f"Epochs: {args.epochs}  |  Batch size: {args.batch_size}  |  LR: {args.lr}")
    print(f"{'='*80}\n")

    model = build_model(num_classes=train_dataset.num_classes, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    history = []
    best_map = -1.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        print(f"\n[Epoch {epoch}/{args.epochs}]")
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_map, _ = evaluate(model, valid_loader, device)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        eta_seconds = (elapsed / epoch) * (args.epochs - epoch)

        is_best = val_map > best_map
        status = " ⭐ NEW BEST!" if is_best else ""

        print(f"\n  Loss: {train_loss:.4f}  |  mAP@0.5: {val_map:.4f}{status}")
        print(f"  Time: {epoch_time:.1f}s  |  Elapsed: {int(elapsed)}s  |  ETA: {int(eta_seconds)}s\n")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_map50": val_map})

        torch.save(
            {"model_state_dict": model.state_dict(), "num_classes": train_dataset.num_classes,
             "categories": train_dataset.categories, "epoch": epoch},
            os.path.join(args.output, "last_model.pth"),
        )
        if is_best:
            best_map = val_map
            torch.save(
                {"model_state_dict": model.state_dict(), "num_classes": train_dataset.num_classes,
                 "categories": train_dataset.categories, "epoch": epoch, "val_map50": val_map},
                os.path.join(args.output, "best_model.pth"),
            )

        with open(os.path.join(args.output, "metrics.json"), "w") as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"✓ Training complete!")
    print(f"  Best val mAP@0.5: {best_map:.4f}")
    print(f"  Total time: {int(total_time)}s ({int(total_time/60)}m)")
    print(f"  Checkpoints saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
