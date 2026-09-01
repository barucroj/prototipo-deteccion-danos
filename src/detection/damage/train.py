"""CLI entrypoint: train the damage detector.

Usage:
    python -m src.detection.damage.train [--data-root DIR] [--output DIR]
        [--epochs N] [--batch-size N] [--lr LR] [--device cpu|cuda]
        [--no-amp] [--max-train-images N]

All the logic lives in :mod:`src.detection.common.trainer`; this file only
supplies the dataset config. Run with ``--help`` for the full flag list.
"""

from src.detection.common import trainer
from src.detection.damage.config import CONFIG

if __name__ == "__main__":
    trainer.main(CONFIG)
