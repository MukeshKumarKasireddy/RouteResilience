"""
Mask cleanup foundation (Part 1).
Implements basic morphological cleanup of binary road masks:
    - thresholding
    - morphological opening (removes small noise specks)
    - morphological closing (fills small gaps/holes)
    - small connected-component removal
This module is intentionally minimal for Part 1. Full occlusion/gap
*recovery* (e.g. reconnecting roads broken by clouds, shadows, or trees)
is out of scope here and will be implemented in Part 2. Keeping cleanup
separate from `inference.py` keeps each module single-purpose and makes
Part 2 a natural extension point rather than a rewrite.
"""

from typing import Optional

import cv2
import numpy as np


def threshold_mask(probability_mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a float probability mask (0-1) into a binary uint8 mask (0/1)."""
    return (probability_mask > threshold).astype(np.uint8)


def morphological_opening(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Remove small isolated noise specks via erosion followed by dilation."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def morphological_closing(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Fill small holes/gaps via dilation followed by erosion."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def remove_small_components(mask: np.ndarray, min_size: int = 30) -> np.ndarray:
    """Remove connected components smaller than `min_size` pixels.
    Helps eliminate spurious tiny blobs that are not meaningful road
    segments, without attempting to reconnect or recover real gaps.
    """
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    cleaned = np.zeros_like(mask_uint8)
    for label_id in range(1, num_labels):  # skip background label 0
        if stats[label_id, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == label_id] = 1
    return cleaned


def clean_mask(
    probability_or_binary_mask: np.ndarray,
    threshold: Optional[float] = 0.5,
    opening_kernel: int = 3,
    closing_kernel: int = 3,
    min_component_size: int = 30,
) -> np.ndarray:
    """Run the full Part-1 cleanup foundation on a mask.
    Args:
        probability_or_binary_mask: either a float probability mask (0-1)
            or an already-binary mask. If `threshold` is None, the input
            is assumed to already be binary (0/1) and thresholding is
            skipped.
        threshold: probability threshold used to binarize; set to None to
            skip (input already binary).
        opening_kernel: structuring element size for morphological opening.
        closing_kernel: structuring element size for morphological closing.
        min_component_size: minimum connected-component pixel area to keep.
    Returns:
        A cleaned binary mask (uint8, values 0/1) of the same shape.
    """
    mask = (
        threshold_mask(probability_or_binary_mask, threshold)
        if threshold is not None
        else probability_or_binary_mask.astype(np.uint8)
    )
    mask = morphological_opening(mask, opening_kernel)
    mask = morphological_closing(mask, closing_kernel)
    mask = remove_small_components(mask, min_component_size)
    return mask