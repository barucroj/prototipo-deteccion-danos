"""CLI entrypoint: run the trained damage detector over images.

Usage:
    python -m src.m3_damage.detector.predict <checkpoint.pth> <input_dir>
        <output_dir> [-n N] [--score-threshold T]

Writes annotated copies for visual inspection, same before/after demo pattern
as ``src/m1_preprocessing/specular_removal.py``. All the logic lives in
:mod:`src.detection_common.predict`; this file only supplies the dataset config.
"""

from src.detection_common import predict
from src.m3_damage.detector.config import CONFIG

if __name__ == "__main__":
    predict.main(CONFIG)
