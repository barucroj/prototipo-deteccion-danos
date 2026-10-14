"""CLI entrypoint: train the damage detector.

Usage:
    python -m src.m3_damage.detector.train [--data-root DIR] [--output DIR]
        [--epochs N] [--batch-size N] [--lr LR] [--device cpu|cuda]
        [--no-amp] [--max-train-images N]

All the logic lives in :mod:`src.detection_common.trainer`; this file only
supplies the dataset config. Run with ``--help`` for the full flag list.
"""

from src.detection_common import trainer
from src.m3_damage.detector.config import CONFIG

if __name__ == "__main__":
    trainer.main(CONFIG)
