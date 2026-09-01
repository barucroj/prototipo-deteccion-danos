"""CLI entrypoint: train the car_parts detector.

Usage:
    python -m src.detection.car_parts.train [--data-root DIR] [--output DIR]
        [--epochs N] [--batch-size N] [--lr LR] [--device cpu|cuda]
        [--no-amp] [--max-train-images N]

All the logic lives in :mod:`src.detection.common.trainer`; this file only
supplies the dataset config. Run with ``--help`` for the full flag list.
"""

from src.detection.common import trainer
from src.detection.car_parts.config import CONFIG

if __name__ == "__main__":
    trainer.main(CONFIG)
