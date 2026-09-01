"""CLI entrypoint: run the trained car_parts detector over images.

Usage:
    python -m src.detection.car_parts.predict <checkpoint.pth> <input_dir>
        <output_dir> [-n N] [--score-threshold T]

Writes annotated copies for visual inspection, same before/after demo pattern
as ``src/preprocessing/specular_removal.py``. All the logic lives in
:mod:`src.detection.common.predict`; this file only supplies the dataset config.
"""

from src.detection.common import predict
from src.detection.car_parts.config import CONFIG

if __name__ == "__main__":
    predict.main(CONFIG)
