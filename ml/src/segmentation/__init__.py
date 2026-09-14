"""Image-level road segmentation pipeline (owned by Suryakant, AI/ML Lead).

Exposes the core building blocks: dataset, model, losses, metrics,
training, and inference. Geospatial/network reconstruction is out of
scope for this package.
"""

from .config import CONFIG, SegmentationConfig
from .dataset import RoadSegmentationDataset
from .model import UNet, build_model

__all__ = [
    "CONFIG",
    "SegmentationConfig",
    "RoadSegmentationDataset",
    "UNet",
    "build_model",
]
