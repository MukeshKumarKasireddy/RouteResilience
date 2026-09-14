"""
Small reusable utilities for the segmentation pipeline: seeding,
device selection, and lightweight logging helpers.
"""

import logging
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds across python, numpy, and torch for reproducibility.

    Deterministic cuDNN settings are also applied where available so that
    repeated runs on the same hardware produce consistent results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available torch device.

    Falls back gracefully to CPU so the pipeline runs anywhere,
    including CI environments without a GPU.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger, safe to call repeatedly (no duplicate handlers)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def resolve_path(path: "os.PathLike[str] | str") -> str:
    """Resolve a path to an absolute string path.

    Centralizing this avoids hard-coded absolute paths scattered through
    the codebase; callers should always pass paths through configuration
    or CLI arguments rather than embedding them directly.
    """
    return os.path.abspath(os.path.expanduser(str(path)))


class MissingFileError(FileNotFoundError):
    """Raised when an expected image or mask file cannot be found."""


class InvalidImageError(ValueError):
    """Raised when an image or mask fails validation (dimensions, format)."""
