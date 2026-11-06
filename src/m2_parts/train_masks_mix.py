"""CLI entrypoint: train the 10-class car_parts Mask R-CNN on Roboflow + HITL (``car_parts/v3_mix``).

Train: Roboflow train (327) + HITL train (802) in one stage; epoch selection:
Roboflow val + HITL val. Same recipe as ``train_masks_10c``; needs the mixed
files of :mod:`src.m2_parts.mix_dataset`. Default output
``models/checkpoints/car_parts/v3_mix``.

Usage:
    python -m src.m2_parts.train_masks_mix --epochs 12 --output DIR ...
"""

from src.detection_common import trainer
from src.m2_parts.config import MASK_MIX_CONFIG

if __name__ == "__main__":
    trainer.main(MASK_MIX_CONFIG)
