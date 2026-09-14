"""Tests for conservative gap scoring, filtering, and recovery.

Uses deterministic synthetic masks only.
"""

import numpy as np
import pytest

from ml.src.postprocessing.gap_detection import CandidatePair, Endpoint, detect_gaps
from ml.src.postprocessing.gap_recovery import (
    GapRecoveryConfig,
    filter_candidates,
    rasterize_connection,
    recover_gaps,
    score_candidate,
)
from ml.src.postprocessing.mask_cleanup import clean_mask


def _aligned_candidate(distance: float = 20.0) -> CandidatePair:
    """A well-aligned, short candidate that should score highly."""
    endpoint_a = Endpoint(x=0, y=0, component_id=1, orientation=0.0, confidence=1.0)
    endpoint_b = Endpoint(
        x=distance, y=0, component_id=2, orientation=np.pi, confidence=1.0
    )
    return CandidatePair(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        distance=distance,
        orientation_diff=0.0,
        direction_diff=0.0,
    )


def _misaligned_candidate(distance: float = 20.0) -> CandidatePair:
    """A perpendicular, poorly-aligned candidate that should be rejected."""
    endpoint_a = Endpoint(x=0, y=0, component_id=1, orientation=0.0, confidence=1.0)
    endpoint_b = Endpoint(
        x=distance, y=0, component_id=2, orientation=np.pi / 2, confidence=1.0
    )
    return CandidatePair(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        distance=distance,
        orientation_diff=np.pi / 2,  # 90 degrees misaligned
        direction_diff=np.pi / 2,
    )


# --- Scoring ---


def test_score_candidate_accepts_well_aligned_short_gap():
    candidate = _aligned_candidate(distance=20.0)
    scored = score_candidate(candidate)
    assert scored.accepted
    assert scored.rejection_reason is None
    assert scored.score > 0.5


def test_score_candidate_rejects_orientation_mismatch():
    candidate = _misaligned_candidate(distance=20.0)
    scored = score_candidate(candidate)
    assert not scored.accepted
    assert scored.rejection_reason in {"orientation_incompatible", "direction_incompatible"}


def test_score_candidate_rejects_distance_beyond_limit():
    config = GapRecoveryConfig(max_gap_distance=10.0)
    candidate = _aligned_candidate(distance=50.0)
    scored = score_candidate(candidate, config=config)
    assert not scored.accepted
    assert scored.rejection_reason == "distance_exceeds_limit"


def test_score_candidate_rejects_low_confidence():
    config = GapRecoveryConfig(min_confidence=0.9)
    endpoint_a = Endpoint(x=0, y=0, component_id=1, orientation=0.0, confidence=0.2)
    endpoint_b = Endpoint(x=20, y=0, component_id=2, orientation=np.pi, confidence=0.2)
    candidate = CandidatePair(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        distance=20.0,
        orientation_diff=0.0,
        direction_diff=0.0,
    )
    scored = score_candidate(candidate, config=config)
    assert not scored.accepted
    assert scored.rejection_reason == "insufficient_confidence"


def test_score_candidate_rejects_score_below_threshold_even_if_gates_pass():
    # Distance close to the limit -> low distance_score, pushing the
    # combined score below a strict min_score even though nothing is
    # individually "incompatible".
    config = GapRecoveryConfig(max_gap_distance=100.0, min_score=0.99)
    candidate = _aligned_candidate(distance=90.0)
    scored = score_candidate(candidate, config=config)
    assert not scored.accepted
    assert scored.rejection_reason == "score_below_threshold"


# --- Filtering / endpoint reuse ---


def test_filter_candidates_splits_accepted_and_rejected():
    candidates = [_aligned_candidate(20.0), _misaligned_candidate(20.0)]
    accepted, rejected = filter_candidates(candidates)
    assert len(accepted) == 1
    assert len(rejected) == 1


def test_filter_candidates_prevents_endpoint_reuse():
    shared_endpoint = Endpoint(x=0, y=0, component_id=1, orientation=0.0, confidence=1.0)
    other_a = Endpoint(x=20, y=0, component_id=2, orientation=np.pi, confidence=1.0)
    other_b = Endpoint(x=20, y=5, component_id=3, orientation=np.pi, confidence=1.0)

    candidate_1 = CandidatePair(
        endpoint_a=shared_endpoint, endpoint_b=other_a,
        distance=20.0, orientation_diff=0.0, direction_diff=0.0,
    )
    candidate_2 = CandidatePair(
        endpoint_a=shared_endpoint, endpoint_b=other_b,
        distance=20.0, orientation_diff=0.0, direction_diff=0.05,
    )
    accepted, rejected = filter_candidates([candidate_1, candidate_2])
    assert len(accepted) == 1
    reused = [r for r in rejected if r.rejection_reason == "endpoint_reused"]
    assert len(reused) == 1


# --- Rasterization ---


def test_rasterize_connection_draws_line_between_endpoints():
    candidate = _aligned_candidate(distance=20.0)
    connection = rasterize_connection((10, 30), candidate, line_width=1)
    assert connection[0, 0] == 1
    assert connection[0, 20] == 1
    assert connection.sum() > 0


# --- Full recovery pipeline ---


def _gapped_mask(gap=40, thickness=3):
    mask = np.zeros((100, 200), dtype=np.uint8)
    half = thickness // 2
    mask[50 - half : 50 + half + 1, 10:80] = 1
    mask[50 - half : 50 + half + 1, 80 + gap : 190] = 1
    return mask


def test_recover_gaps_accepts_a_plausible_aligned_gap():
    mask = _gapped_mask(gap=40)
    cleaned = clean_mask(mask, threshold=None, min_component_size=1)
    result = recover_gaps(cleaned)
    assert len(result.accepted) == 1
    assert result.recovered_pixels_mask.sum() > 0
    assert result.average_gap_length > 0


def test_recover_gaps_rejects_gap_beyond_configured_distance():
    # A gap that is geometrically plausible (well-aligned) but exceeds a
    # strict configured max_distance must still be rejected.
    mask = _gapped_mask(gap=40)
    cleaned = clean_mask(mask, threshold=None, min_component_size=1)
    config = GapRecoveryConfig(max_gap_distance=5.0)
    result = recover_gaps(cleaned, config=config)
    assert len(result.accepted) == 0
    assert result.recovered_pixels_mask.sum() == 0


def test_recover_gaps_preserves_original_road_pixels_when_pre_cleaned():
    mask = _gapped_mask(gap=40)
    cleaned = clean_mask(mask, threshold=None, min_component_size=1)
    result = recover_gaps(cleaned)
    # Every pixel that was part of the cleaned input mask must still be
    # present in the final recovered mask (recovery only adds pixels).
    assert np.all(result.recovered_mask[cleaned > 0] > 0)


def test_recover_gaps_output_mask_has_same_dimensions_as_input():
    mask = _gapped_mask(gap=40)
    cleaned = clean_mask(mask, threshold=None, min_component_size=1)
    result = recover_gaps(cleaned)
    assert result.recovered_mask.shape == cleaned.shape
    assert result.recovered_pixels_mask.shape == cleaned.shape


def test_recover_gaps_no_recovery_case_single_component():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 1  # single connected component, nothing to recover
    # Skip the final cleanup pass here to assert exact preservation; final
    # cleanup on a *non-pre-cleaned* mask can trim a few corner pixels via
    # normal morphological opening, which is expected (see
    # gap_recovery.recover_gaps docstring) and covered separately by
    # test_recover_gaps_preserves_original_road_pixels_when_pre_cleaned.
    result = recover_gaps(mask, apply_final_cleanup=False)
    assert len(result.accepted) == 0
    assert result.recovered_pixels_mask.sum() == 0
    assert np.array_equal(result.recovered_mask > 0, mask > 0)


def test_recover_gaps_uses_confidence_from_probability_map():
    mask = _gapped_mask(gap=40)
    cleaned = clean_mask(mask, threshold=None, min_component_size=1)

    low_conf_map = np.full(cleaned.shape, 0.05, dtype=np.float32)
    config = GapRecoveryConfig(min_confidence=0.5)
    result = recover_gaps(cleaned, probability_map=low_conf_map, config=config)
    # Low confidence everywhere -> the otherwise-plausible gap must be rejected.
    assert len(result.accepted) == 0
