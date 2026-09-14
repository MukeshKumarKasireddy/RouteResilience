"""Additional tests for the mask cleanup foundation, covering edge cases
relevant to Phase 2 integration (empty masks, already-binary masks).
"""

import numpy as np

from ml.src.postprocessing.mask_cleanup import (
    clean_mask,
    morphological_closing,
    morphological_opening,
    remove_small_components,
    threshold_mask,
)


def test_threshold_mask_basic():
    prob = np.array([[0.1, 0.9], [0.4, 0.6]])
    binary = threshold_mask(prob, threshold=0.5)
    assert binary.tolist() == [[0, 1], [0, 1]]


def test_clean_mask_on_empty_mask_returns_empty():
    empty = np.zeros((32, 32), dtype=np.uint8)
    cleaned = clean_mask(empty, threshold=None)
    assert cleaned.sum() == 0
    assert cleaned.shape == empty.shape


def test_clean_mask_removes_small_noise_speck():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5, 5] = 1  # single-pixel noise
    mask[10:20, 10:20] = 1  # a real 10x10 blob
    cleaned = clean_mask(mask, threshold=None, min_component_size=5)
    assert cleaned[5, 5] == 0
    assert cleaned[15, 15] == 1


def test_remove_small_components_keeps_large_ones():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0, 0] = 1  # isolated pixel, area 1
    mask[5:15, 5:15] = 1  # area 100
    cleaned = remove_small_components(mask, min_size=10)
    assert cleaned[0, 0] == 0
    assert cleaned[10, 10] == 1


def test_morphological_opening_and_closing_are_shape_preserving():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    opened = morphological_opening(mask, kernel_size=3)
    closed = morphological_closing(opened, kernel_size=3)
    assert opened.shape == mask.shape
    assert closed.shape == mask.shape
