"""CLI entrypoint: train the 10-class car_parts Mask R-CNN on Roboflow only (``car_parts/v3_10c``).

Same shared trainer and recipe as ``train_masks.py`` (``car_parts/v2``). Train:
Roboflow train; epoch selection: Roboflow val + HITL val (the mixed files of
:mod:`src.m2_parts.mix_dataset`, which must exist). Counterpart of
``train_masks_mix`` (``car_parts/v3_mix``), which differs only in the
training images. Default output ``models/checkpoints/car_parts/v3_10c``.

Usage:
    python -m src.m2_parts.train_masks_10c --epochs 12 --output DIR ...
"""

from src.detection_common import trainer
from src.m2_parts.config import MASK_10_CONFIG

if __name__ == "__main__":
    trainer.main(MASK_10_CONFIG)
