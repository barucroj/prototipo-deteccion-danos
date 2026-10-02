"""CLI entrypoint: train the car_parts detector WITH masks (Mask R-CNN).

This is the model module M3 needs as input: per image, a list of
``{part class, score, box, mask}`` whose masks are intersected with each
damage mask to assign damages to parts.

Usage:
    python -m src.detection.car_parts.train_masks [--epochs N]
        [--lr-step-size N] [--output DIR] [--seed N] [--no-augment] ...

Same shared trainer as ``train.py`` (boxes only, ``car_parts/v1``); only the
config differs: ``with_masks=True``, ``arch="mask_rcnn"``, default output
``models/checkpoints/car_parts/v2`` and ``best_model.pth`` selected by mask
AP. Run with ``--help`` for the full flag list.

Progress while it runs, from another terminal:
    python -m src.detection.common.watch models/checkpoints/car_parts/v2
"""

from src.detection.common import trainer
from src.detection.car_parts.config import MASK_CONFIG

if __name__ == "__main__":
    trainer.main(MASK_CONFIG)
