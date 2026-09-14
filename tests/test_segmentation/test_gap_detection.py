"""Tests for gap detection: connected components, skeletonization,
endpoint detection, and candidate pairing. Uses only deterministic
synthetic masks -- no real satellite data.
"""

import numpy as np
import pytest

from ml.src.postprocessing.gap_detection import (
    Endpoint,
    InvalidMaskError,
    detect_connected_components,
    detect_endpoints,
    detect_gaps,
    find_candidate_pairs,
    skeletonize_mask,
    validate_binary_mask,
)


def _two_segment_mask(shape=(100, 200), gap_start=80, gap_end=120, row=50, thickness=3):
    mask = np.zeros(shape, dtype=np.uint8)
    half = thickness // 2
    mask[row - half : row + half + 1, 10:gap_start] = 1
    mask[row - half : row + half + 1, gap_end:190] = 1
    return mask


# --- Mask validation ---


def test_validate_binary_mask_accepts_0_1():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    validated = validate_binary_mask(mask)
    assert set(np.unique(validated).tolist()) <= {0, 1}


def test_validate_binary_mask_accepts_0_255():
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    validated = validate_binary_mask(mask)
    assert set(np.unique(validated).tolist()) <= {0, 1}


def test_validate_binary_mask_rejects_non_binary_values():
    mask = np.array([[0, 128], [64, 255]], dtype=np.uint8)
    with pytest.raises(InvalidMaskError):
        validate_binary_mask(mask)


def test_validate_binary_mask_rejects_empty_mask():
    with pytest.raises(InvalidMaskError):
        validate_binary_mask(np.zeros((0, 0), dtype=np.uint8))


def test_validate_binary_mask_rejects_non_2d():
    with pytest.raises(InvalidMaskError):
        validate_binary_mask(np.zeros((4, 4, 3), dtype=np.uint8))


def test_validate_binary_mask_rejects_non_array():
    with pytest.raises(InvalidMaskError):
        validate_binary_mask([[0, 1], [1, 0]])


# --- Connected components ---


def test_detect_connected_components_single_component():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    num_components, labels, stats = detect_connected_components(mask)
    assert num_components == 1
    assert labels.shape == mask.shape


def test_detect_connected_components_multiple_disconnected():
    mask = _two_segment_mask()
    num_components, labels, stats = detect_connected_components(mask)
    assert num_components == 2


def test_detect_connected_components_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    num_components, labels, stats = detect_connected_components(mask)
    assert num_components == 0


# --- Skeletonization ---


def test_skeletonize_mask_thins_thick_road():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1  # a thick 10x10 blob
    skeleton = skeletonize_mask(mask)
    assert skeleton.sum() < mask.sum()
    assert skeleton.shape == mask.shape


def test_skeletonize_empty_mask_stays_empty():
    mask = np.zeros((10, 10), dtype=np.uint8)
    skeleton = skeletonize_mask(mask)
    assert skeleton.sum() == 0


# --- Endpoint detection ---


def test_detect_endpoints_finds_two_endpoints_per_line_segment():
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[10, 5:35] = 1  # a single-pixel-wide straight line
    skeleton = skeletonize_mask(mask)
    _, labels, _ = detect_connected_components(mask)
    endpoints = detect_endpoints(skeleton, labels)
    assert len(endpoints) == 2
    xs = sorted(e.x for e in endpoints)
    assert xs[0] <= 6 and xs[1] >= 34


def test_detect_endpoints_handles_isolated_pixel_without_crashing():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 1  # single isolated pixel, no neighbors
    skeleton = skeletonize_mask(mask)
    _, labels, _ = detect_connected_components(mask)
    endpoints = detect_endpoints(skeleton, labels)
    # An isolated pixel has 0 skeleton neighbors -> not counted as an endpoint.
    assert endpoints == []


def test_detect_endpoints_on_noisy_skeleton_does_not_crash():
    rng = np.random.default_rng(0)
    mask = (rng.random((30, 30)) > 0.7).astype(np.uint8)  # random noise mask
    skeleton = skeletonize_mask(mask)
    _, labels, _ = detect_connected_components(mask)
    # Should complete without raising, regardless of how noisy the input is.
    endpoints = detect_endpoints(skeleton, labels)
    assert isinstance(endpoints, list)


def test_detect_endpoints_component_with_no_endpoints_eg_closed_loop():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    mask[7:13, 7:13] = 0  # hollow it out -> a ring/loop shape
    skeleton = skeletonize_mask(mask)
    _, labels, _ = detect_connected_components(mask)
    # Should not crash; a closed loop skeleton may have zero true endpoints.
    endpoints = detect_endpoints(skeleton, labels)
    assert isinstance(endpoints, list)


def test_detect_endpoints_uses_probability_map_for_confidence():
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[10, 5:35] = 1
    skeleton = skeletonize_mask(mask)
    _, labels, _ = detect_connected_components(mask)

    prob_map = np.full((20, 40), 0.9, dtype=np.float32)
    endpoints = detect_endpoints(skeleton, labels, probability_map=prob_map)
    assert len(endpoints) == 2
    for e in endpoints:
        assert 0.0 <= e.confidence <= 1.0
        assert e.confidence > 0.5  # neighborhood is uniformly high-confidence


# --- Candidate pairing ---


def test_find_candidate_pairs_only_pairs_different_components():
    e1 = Endpoint(x=0, y=0, component_id=1, orientation=0.0)
    e2 = Endpoint(x=5, y=0, component_id=1, orientation=0.0)  # same component
    e3 = Endpoint(x=10, y=0, component_id=2, orientation=3.14159)
    candidates = find_candidate_pairs([e1, e2, e3], max_distance=100)
    component_pairs = {
        (c.endpoint_a.component_id, c.endpoint_b.component_id) for c in candidates
    }
    assert (1, 1) not in component_pairs


def test_find_candidate_pairs_rejects_beyond_max_distance():
    e1 = Endpoint(x=0, y=0, component_id=1, orientation=0.0)
    e2 = Endpoint(x=100, y=0, component_id=2, orientation=3.14159)
    candidates = find_candidate_pairs([e1, e2], max_distance=10)
    assert candidates == []


def test_find_candidate_pairs_accepts_within_max_distance():
    e1 = Endpoint(x=0, y=0, component_id=1, orientation=0.0)
    e2 = Endpoint(x=5, y=0, component_id=2, orientation=3.14159)
    candidates = find_candidate_pairs([e1, e2], max_distance=10)
    assert len(candidates) == 1
    assert candidates[0].distance == pytest.approx(5.0)


# --- Full detect_gaps pipeline ---


def test_detect_gaps_end_to_end_finds_the_gap():
    mask = _two_segment_mask()
    result = detect_gaps(mask, max_distance=60)
    assert result.num_components == 2
    assert len(result.candidates) == 1
    assert result.candidates[0].distance == pytest.approx(41.0, abs=2.0)


def test_detect_gaps_on_single_component_returns_no_candidates():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    result = detect_gaps(mask)
    assert result.num_components == 1
    assert result.candidates == []


def test_detect_gaps_on_empty_mask_returns_no_candidates():
    mask = np.zeros((20, 20), dtype=np.uint8)
    result = detect_gaps(mask)
    assert result.num_components == 0
    assert result.endpoints == []
    assert result.candidates == []
