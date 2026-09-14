"""
Centralized configuration for the road segmentation pipeline.

All tunable constants for dataset loading, training, and inference live
here so they are not scattered across modules. Values can be overridden
via environment variables for quick experimentation without editing code.

Ownership: Suryakant (AI/ML Lead) - image-level segmentation pipeline only.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Repository root, resolved relative to this file so the code works
# regardless of the current working directory it is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SegmentationConfig:
    """Configuration for U-Net road segmentation training and inference."""

    # --- Data ---
    image_size: int = _env_int("SEG_IMAGE_SIZE", 256)
    in_channels: int = _env_int("SEG_IN_CHANNELS", 3)
    out_channels: int = _env_int("SEG_OUT_CHANNELS", 1)

    data_root: Path = field(default_factory=lambda: REPO_ROOT / "ml" / "data")
    samples_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "ml" / "data" / "samples"
    )
    processed_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "ml" / "data" / "processed"
    )

    # --- Training ---
    batch_size: int = _env_int("SEG_BATCH_SIZE", 8)
    num_epochs: int = _env_int("SEG_NUM_EPOCHS", 20)
    learning_rate: float = _env_float("SEG_LR", 1e-3)
    weight_decay: float = _env_float("SEG_WEIGHT_DECAY", 1e-5)
    num_workers: int = _env_int("SEG_NUM_WORKERS", 2)
    validation_split: float = _env_float("SEG_VAL_SPLIT", 0.2)
    random_seed: int = _env_int("SEG_SEED", 42)

    # --- Inference / thresholding ---
    threshold: float = _env_float("SEG_THRESHOLD", 0.5)

    # --- Paths ---
    model_dir: Path = field(default_factory=lambda: REPO_ROOT / "ml" / "models")
    model_filename: str = _env_str("SEG_MODEL_FILENAME", "unet_road_segmentation.pth")
    output_dir: Path = field(default_factory=lambda: REPO_ROOT / "ml" / "outputs")
    predictions_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "outputs" / "predictions"
    )

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename

    def ensure_dirs(self) -> None:
        """Create output directories if they do not already exist."""
        for path in (self.model_dir, self.output_dir, self.predictions_dir):
            path.mkdir(parents=True, exist_ok=True)


# Single shared config instance used across the segmentation package.
CONFIG = SegmentationConfig()
