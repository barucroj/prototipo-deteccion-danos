"""Shared training driver: one CLI and one training loop for both detectors.

``src/detection/car_parts/train.py`` and ``src/detection/damage/train.py`` are
thin wrappers that pass their :class:`DetectorConfig` in here, so the two
models cannot drift apart in optimizer settings, checkpoint format or
reported metrics.

Checkpoints are self-describing: they carry ``num_classes``, ``arch``,
``with_masks`` and the ``categories`` id->name map, so ``predict.py`` can
rebuild the right model without being told which detector it came from.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from src.detection.common.coco_dataset import build_dataset, collate_fn
from src.detection.common.coco_eval import PYCOCOTOOLS_AVAILABLE, evaluate_coco
from src.detection.common.model import ARCHITECTURES, build_model
from src.detection.common.transforms import build_train_transforms


def build_arg_parser(cfg) -> argparse.ArgumentParser:
    """CLI shared by both train entrypoints, with per-dataset defaults."""
    parser = argparse.ArgumentParser(description=f"Train the {cfg.description}.")
    parser.add_argument("--data-root", default=cfg.data_root,
                        help="Dataset root containing the split folders.")
    parser.add_argument("--output", default=cfg.default_output,
                        help="Folder to write checkpoints and metrics.json to.")
    parser.add_argument("--epochs", type=int, default=cfg.default_epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.default_batch_size)
    parser.add_argument("--lr", type=float, default=cfg.default_lr)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--arch", default=cfg.arch, choices=ARCHITECTURES)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", dest="amp", action="store_true", default=None,
                        help="Force mixed precision on (default: on when device is cuda).")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Force mixed precision off.")
    parser.add_argument("--augment", dest="augment", action="store_true", default=True,
                        help="Train-time augmentation: horizontal flip plus brightness, "
                             "contrast and saturation jitter. On by default; never applied "
                             "to the validation split.")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="Disable train-time augmentation.")
    parser.add_argument("--hflip-prob", type=float, default=0.5,
                        help="Probability of the horizontal flip (default 0.5).")
    parser.add_argument("--jitter-prob", type=float, default=0.5,
                        help="Probability of the photometric jitter (default 0.5).")
    parser.add_argument("--metric", default="coco" if PYCOCOTOOLS_AVAILABLE else "simple",
                        choices=("coco", "simple"),
                        help="'coco' = standard COCO AP via pycocotools (comparable to "
                             "published numbers; the one to quote). 'simple' = the "
                             "from-scratch mAP@0.5 fallback. The two are NOT comparable.")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="Evaluate every N epochs (the final epoch is always "
                             "evaluated). Validation is ~20%% of a CarDD epoch, so 2 "
                             "meaningfully shortens a long run at the cost of a coarser curve.")
    parser.add_argument("--lr-step-size", type=int, default=None,
                        help="If set, StepLR every N epochs (gamma 0.1).")
    parser.add_argument("--max-train-images", type=int, default=None,
                        help="If set, cap the training set to this many images (smoke tests).")
    parser.add_argument("--max-val-images", type=int, default=None,
                        help="If set, cap the validation set to this many images (smoke tests). "
                             "mAP from a capped split is not comparable to a full-split run.")
    return parser


def run_training(cfg, args) -> float:
    """Train, evaluating on the validation split after every epoch.

    Writes ``best_model.pth`` (highest validation mAP so far),
    ``last_model.pth`` and ``metrics.json`` to ``args.output``.

    Returns:
        The best validation mAP@0.5 reached.
    """
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device)
    use_amp = args.amp if args.amp is not None else (device.type == "cuda")

    if args.metric == "coco" and not PYCOCOTOOLS_AVAILABLE:
        raise SystemExit("--metric coco requires pycocotools; install it or pass --metric simple")

    # Augmentation goes on the training split only: augmenting validation would
    # make its scores incomparable between runs.
    train_transforms = build_train_transforms(
        hflip_prob=args.hflip_prob, jitter_prob=args.jitter_prob
    ) if args.augment else None

    train_dataset = build_dataset(cfg, cfg.train_split, args.data_root,
                                  transforms=train_transforms)
    valid_dataset = build_dataset(cfg, cfg.val_split, args.data_root)

    if args.max_train_images is not None:
        train_dataset.samples = train_dataset.samples[: args.max_train_images]
    if args.max_val_images is not None:
        valid_dataset.samples = valid_dataset.samples[: args.max_val_images]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn)

    print(f"\n{'='*80}")
    print(f"Detector: {cfg.name} — {cfg.description}")
    print(f"Train images: {len(train_dataset)}  |  Valid images: {len(valid_dataset)}")
    print(f"Classes: {train_dataset.num_classes}  |  Arch: {args.arch}  |  Masks: {cfg.with_masks}")
    print(f"Device: {device}  |  AMP: {use_amp}  |  Metric: {args.metric}")
    print(f"Augmentation: {train_transforms if train_transforms else 'off'}")
    print(f"Epochs: {args.epochs}  |  Batch size: {args.batch_size}  |  LR: {args.lr}")
    print(f"{'='*80}\n")

    model = build_model(train_dataset.num_classes, arch=args.arch, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)
        if args.lr_step_size else None
    )
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    # Imported here so the module stays importable without the heavier deps loaded early.
    from src.detection.common.engine import evaluate, train_one_epoch

    def checkpoint_payload(epoch, val_map=None, val_map50=None, coco_stats=None):
        payload = {
            "model_state_dict": model.state_dict(),
            "num_classes": train_dataset.num_classes,
            "categories": train_dataset.categories,
            "arch": args.arch,
            "with_masks": cfg.with_masks,
            "detector": cfg.name,
            "metric": args.metric,
            "augmented": bool(train_transforms),
            "epoch": epoch,
        }
        # Under --metric coco, val_map is AP@0.5:0.95 and val_map50 is AP50;
        # under --metric simple both are the same from-scratch mAP@0.5.
        if val_map is not None:
            payload["val_map"] = val_map
            payload["val_map50"] = val_map50
        if coco_stats:
            payload["coco"] = coco_stats
        return payload

    history = []
    best_map = -1.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n[Epoch {epoch}/{args.epochs}]")

        train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler=scaler)

        is_eval_epoch = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        val_map = val_map50 = None
        coco_stats = {}
        per_class_ap = {}

        if is_eval_epoch:
            if args.metric == "coco":
                coco_stats = evaluate_coco(
                    model, valid_loader, device, valid_dataset.ann_json_path,
                    image_ids=valid_dataset.coco_image_ids, with_masks=cfg.with_masks,
                )
                # Checkpoint selection follows the primary COCO metric, AP@0.5:0.95.
                val_map = coco_stats["bbox"]["AP"] if coco_stats else 0.0
                val_map50 = coco_stats["bbox"]["AP50"] if coco_stats else 0.0
            else:
                val_map50, per_class_ap = evaluate(model, valid_loader, device)
                val_map = val_map50

        if lr_scheduler is not None:
            lr_scheduler.step()

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        eta_seconds = (elapsed / epoch) * (args.epochs - epoch)

        is_best = val_map is not None and val_map > best_map
        status = " * NEW BEST!" if is_best else ""

        if val_map is None:
            map_text = "eval skipped"
        elif args.metric == "coco":
            map_text = f"AP {val_map:.4f} | AP50 {val_map50:.4f}"
            if coco_stats.get("segm"):
                map_text += f" | mask AP {coco_stats['segm']['AP']:.4f}"
            map_text += status
        else:
            map_text = f"mAP@0.5 {val_map:.4f}{status}"

        print(f"\n  Loss: {train_loss:.4f}  |  {map_text}")
        print(f"  Time: {epoch_time:.1f}s  |  Elapsed: {int(elapsed)}s  |  ETA: {int(eta_seconds)}s\n")

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "metric": args.metric,
            "val_map": val_map,      # COCO AP@0.5:0.95, or the simple mAP@0.5 fallback
            "val_map50": val_map50,
        }
        if coco_stats:
            record["coco"] = coco_stats
        if per_class_ap:
            record["per_class_ap"] = {str(k): v for k, v in sorted(per_class_ap.items())}
        history.append(record)

        torch.save(checkpoint_payload(epoch, val_map, val_map50, coco_stats),
                   os.path.join(args.output, "last_model.pth"))
        if is_best:
            best_map = val_map
            torch.save(checkpoint_payload(epoch, val_map, val_map50, coco_stats),
                       os.path.join(args.output, "best_model.pth"))

        with open(os.path.join(args.output, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("Training complete.")
    metric_label = "COCO AP@0.5:0.95" if args.metric == "coco" else "mAP@0.5 (simple)"
    print(f"  Best val {metric_label}: {best_map:.4f}")
    print(f"  Total time: {int(total_time)}s ({int(total_time/60)}m)")
    print(f"  Checkpoints saved to: {args.output}")
    print(f"{'='*80}\n")
    return best_map


def main(cfg):
    """Entrypoint used by both ``train.py`` wrappers."""
    run_training(cfg, build_arg_parser(cfg).parse_args())
