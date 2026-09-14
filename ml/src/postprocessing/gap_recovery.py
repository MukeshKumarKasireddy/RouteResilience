"""
Conservative gap recovery for cleaned binary road masks (Phase 2).

Takes the candidate gaps produced by `gap_detection.py` and decides,
conservatively and explainably, which ones represent enough evidence to be
recovered as an actual road connection. Accepted connections are then
rasterized and merged back into the mask.

Design principle (see docs/research/segmentation.md for the full
rationale): **prefer false negatives over inventing roads**. A gap is only
recovered when distance, orientation, direction, and confidence evidence
all clear configurable thresholds and the combined score exceeds a
configurable minimum. Nothing here uses unexplained magic numbers -- every
threshold and weight lives in `GapRecoveryConfig`.

Ownership: Suryakant (AI/ML Lead) - image-level gap recovery only. This
module does not perform geospatial vectorization, CRS handling, or graph
construction.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .gap_detection import (
    CandidatePair,
    GapDetectionResult,
    detect_gaps,
)
from .mask_cleanup import clean_mask


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


@dataclass(frozen=True)
class GapRecoveryConfig:
    """All tunable thresholds/weights for gap detection + recovery.

    Centralized here (mirroring the convention in
    `ml/src/segmentation/config.py`) so nothing is a scattered magic
    number. Every value can be overridden via environment variable for
    quick experimentation.
    """

    # --- Detection ---
    max_gap_distance: float = _env_float("GAP_MAX_DISTANCE", 60.0)
    orientation_radius: int = int(_env_float("GAP_ORIENTATION_RADIUS", 7))
    confidence_radius: int = int(_env_float("GAP_CONFIDENCE_RADIUS", 3))

    # --- Conservative acceptance thresholds ---
    # Reject candidates whose road-direction misalignment exceeds this
    # (radians). ~51 degrees: the two road segments must be roughly
    # collinear, not perpendicular or reversed.
    max_orientation_diff: float = _env_float("GAP_MAX_ORIENTATION_DIFF", 0.9)
    # Reject candidates where the connecting line deviates too far from
    # each endpoint's own outward direction (radians, ~46 degrees).
    max_direction_diff: float = _env_float("GAP_MAX_DIRECTION_DIFF", 0.8)
    # Minimum mean confidence (from the probability map, if supplied)
    # required at both endpoints for a connection to even be considered.
    min_confidence: float = _env_float("GAP_MIN_CONFIDENCE", 0.3)
    # Final combined score (0-1) a candidate must exceed to be accepted.
    min_score: float = _env_float("GAP_MIN_SCORE", 0.55)

    # --- Scoring weights (should sum to 1.0; not enforced, but documented) ---
    weight_distance: float = _env_float("GAP_WEIGHT_DISTANCE", 0.3)
    weight_orientation: float = _env_float("GAP_WEIGHT_ORIENTATION", 0.3)
    weight_direction: float = _env_float("GAP_WEIGHT_DIRECTION", 0.25)
    weight_confidence: float = _env_float("GAP_WEIGHT_CONFIDENCE", 0.15)

    # --- Rasterization ---
    # Width (pixels) of the line drawn for a recovered connection.
    recovered_line_width: int = int(_env_float("GAP_LINE_WIDTH", 2))

    # --- Final cleanup (reuses mask_cleanup.clean_mask on the merged mask) ---
    final_opening_kernel: int = int(_env_float("GAP_FINAL_OPENING_KERNEL", 3))
    final_closing_kernel: int = int(_env_float("GAP_FINAL_CLOSING_KERNEL", 3))
    final_min_component_size: int = int(_env_float("GAP_FINAL_MIN_COMPONENT_SIZE", 30))


CONFIG = GapRecoveryConfig()


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate gap with its computed sub-scores and acceptance decision."""

    candidate: CandidatePair
    distance_score: float
    orientation_score: float
    direction_score: float
    confidence_score: float
    score: float
    accepted: bool
    rejection_reason: Optional[str] = None


def _linear_score(value: float, good_at: float, bad_at: float) -> float:
    """Linearly map `value` to a 0-1 score: 1.0 at `good_at`, 0.0 at `bad_at`.

    `good_at` may be less than or greater than `bad_at` (the function
    handles both increasing and decreasing scores). Values beyond `bad_at`
    clamp to 0.0; values at/beyond `good_at` clamp to 1.0.
    """
    if good_at == bad_at:
        return 1.0 if value == good_at else 0.0
    fraction = (value - bad_at) / (good_at - bad_at)
    return float(np.clip(fraction, 0.0, 1.0))


def score_candidate(
    candidate: CandidatePair,
    config: GapRecoveryConfig = CONFIG,
) -> ScoredCandidate:
    """Compute an interpretable, weighted acceptance score for a candidate gap.

    Sub-scores (each in [0, 1], higher = better evidence for a real gap):

    - **distance_score**: shorter gaps are more plausible than longer ones.
      1.0 at distance 0, 0.0 at `config.max_gap_distance`.
    - **orientation_score**: how collinear the two road segments are.
      1.0 when perfectly aligned (orientation_diff = 0), 0.0 at
      `config.max_orientation_diff`.
    - **direction_score**: how well the straight connecting line continues
      each endpoint's own local road direction. 1.0 when perfectly
      continuous (direction_diff = 0), 0.0 at `config.max_direction_diff`.
    - **confidence_score**: mean of both endpoints' probability-map
      confidence (1.0 if no probability map was available).

    combined score = weighted sum of the four sub-scores (weights from
    `config`).

    A candidate is `accepted` only if *all* hard thresholds are cleared
    (distance, orientation, direction, confidence) *and* the combined score
    exceeds `config.min_score`. This two-stage design (hard gates + soft
    score) is what keeps recovery conservative: a candidate cannot "buy
    back" an implausible orientation with a very short distance, for
    example -- every signal must independently clear its own bar.
    """
    rejection_reason: Optional[str] = None

    distance_score = _linear_score(
        candidate.distance, good_at=0.0, bad_at=config.max_gap_distance
    )
    orientation_score = _linear_score(
        candidate.orientation_diff, good_at=0.0, bad_at=config.max_orientation_diff
    )
    direction_score = _linear_score(
        candidate.direction_diff, good_at=0.0, bad_at=config.max_direction_diff
    )
    confidence_score = float(
        np.clip(
            (candidate.endpoint_a.confidence + candidate.endpoint_b.confidence) / 2.0,
            0.0,
            1.0,
        )
    )

    combined_score = (
        config.weight_distance * distance_score
        + config.weight_orientation * orientation_score
        + config.weight_direction * direction_score
        + config.weight_confidence * confidence_score
    )

    if candidate.distance > config.max_gap_distance:
        rejection_reason = "distance_exceeds_limit"
    elif candidate.orientation_diff > config.max_orientation_diff:
        rejection_reason = "orientation_incompatible"
    elif candidate.direction_diff > config.max_direction_diff:
        rejection_reason = "direction_incompatible"
    elif confidence_score < config.min_confidence:
        rejection_reason = "insufficient_confidence"
    elif combined_score < config.min_score:
        rejection_reason = "score_below_threshold"

    accepted = rejection_reason is None

    return ScoredCandidate(
        candidate=candidate,
        distance_score=distance_score,
        orientation_score=orientation_score,
        direction_score=direction_score,
        confidence_score=confidence_score,
        score=float(combined_score),
        accepted=accepted,
        rejection_reason=rejection_reason,
    )


def filter_candidates(
    candidates: List[CandidatePair],
    config: GapRecoveryConfig = CONFIG,
) -> Tuple[List[ScoredCandidate], List[ScoredCandidate]]:
    """Score every candidate and split into (accepted, rejected) lists.

    One candidate endpoint can only be used once: if multiple accepted
    candidates share an endpoint, only the highest-scoring one is kept, and
    the rest are demoted to rejected (`rejection_reason="endpoint_reused"`).
    This avoids fan-out artifacts (one road tip "recovering" into several
    different directions at once).
    """
    scored = [score_candidate(c, config) for c in candidates]
    accepted = [s for s in scored if s.accepted]
    rejected = [s for s in scored if not s.accepted]

    accepted.sort(key=lambda s: s.score, reverse=True)
    used_endpoints = set()
    final_accepted: List[ScoredCandidate] = []
    for scored_candidate in accepted:
        key_a = (scored_candidate.candidate.endpoint_a.component_id,
                  scored_candidate.candidate.endpoint_a.x,
                  scored_candidate.candidate.endpoint_a.y)
        key_b = (scored_candidate.candidate.endpoint_b.component_id,
                  scored_candidate.candidate.endpoint_b.x,
                  scored_candidate.candidate.endpoint_b.y)
        if key_a in used_endpoints or key_b in used_endpoints:
            rejected.append(
                ScoredCandidate(
                    candidate=scored_candidate.candidate,
                    distance_score=scored_candidate.distance_score,
                    orientation_score=scored_candidate.orientation_score,
                    direction_score=scored_candidate.direction_score,
                    confidence_score=scored_candidate.confidence_score,
                    score=scored_candidate.score,
                    accepted=False,
                    rejection_reason="endpoint_reused",
                )
            )
            continue
        used_endpoints.add(key_a)
        used_endpoints.add(key_b)
        final_accepted.append(scored_candidate)

    return final_accepted, rejected


def rasterize_connection(
    shape: Tuple[int, int],
    candidate: CandidatePair,
    line_width: int = 2,
) -> np.ndarray:
    """Rasterize a single accepted gap connection as a straight-line mask.

    Straight-line interpolation is used deliberately as the simplest robust
    method for Phase 2: real road curvature over a short recovered gap
    (bounded by `max_gap_distance`) is well approximated by a straight
    segment, and a straight line has no free parameters that could
    introduce unjustified curvature. Returns a uint8 mask (0/1) of `shape`
    with only the new connection pixels set.
    """
    connection_mask = np.zeros(shape, dtype=np.uint8)
    point_a = (int(round(candidate.endpoint_a.x)), int(round(candidate.endpoint_a.y)))
    point_b = (int(round(candidate.endpoint_b.x)), int(round(candidate.endpoint_b.y)))
    cv2.line(connection_mask, point_a, point_b, color=1, thickness=line_width)
    return connection_mask


@dataclass
class GapRecoveryResult:
    """Full output of the gap-recovery stage."""

    detection: GapDetectionResult
    accepted: List[ScoredCandidate]
    rejected: List[ScoredCandidate]
    recovered_pixels_mask: np.ndarray  # only the newly-added pixels (0/1)
    recovered_mask: np.ndarray  # cleaned_mask OR recovered_pixels_mask, cleaned
    average_gap_length: float


def recover_gaps(
    cleaned_mask: np.ndarray,
    probability_map: Optional[np.ndarray] = None,
    config: GapRecoveryConfig = CONFIG,
    apply_final_cleanup: bool = True,
) -> GapRecoveryResult:
    """Run full Phase 2 recovery on an already-cleaned binary road mask.

    Pipeline: detect gaps -> score & conservatively filter candidates ->
    rasterize accepted connections -> merge with the original mask ->
    (optional) final cleanup pass.

    Original road pixels are preserved by construction during the merge
    step: recovery only ever adds pixels (`np.maximum`/logical-OR with the
    input mask), never subtracts. `cleaned_mask` is expected to have
    already been through `mask_cleanup.clean_mask` once (as it is in
    `pipeline.run_pipeline`); because morphological opening/closing are
    idempotent, re-running the same cleanup as the optional final pass will
    not further erode a mask that already passed through it. Passing an
    *uncleaned* mask directly, or a different kernel size than was used
    upstream, can still cause the final cleanup pass to trim a small number
    of boundary/corner pixels (a normal morphological opening effect) --
    call with `apply_final_cleanup=False` if byte-exact preservation of a
    non-pre-cleaned mask's pixels is required.

    Args:
        cleaned_mask: binary road mask, already passed through
            `mask_cleanup.clean_mask` (0/1 or 0/255).
        probability_map: optional float probability mask (0-1) used as
            supporting confidence evidence during scoring.
        config: recovery thresholds/weights.
        apply_final_cleanup: whether to run a final morphological cleanup
            pass over the merged (original + recovered) mask.

    Returns:
        A `GapRecoveryResult` with the full audit trail (detection results,
        accepted/rejected candidates with scores, and both the raw
        recovered-pixels mask and final merged+cleaned mask).
    """
    detection = detect_gaps(
        cleaned_mask,
        probability_map=probability_map,
        max_distance=config.max_gap_distance,
        orientation_radius=config.orientation_radius,
        confidence_radius=config.confidence_radius,
    )

    accepted, rejected = filter_candidates(detection.candidates, config=config)

    recovered_pixels = np.zeros_like(detection.binary_mask)
    for scored_candidate in accepted:
        connection = rasterize_connection(
            detection.binary_mask.shape,
            scored_candidate.candidate,
            line_width=config.recovered_line_width,
        )
        recovered_pixels = np.maximum(recovered_pixels, connection)

    # Preserve original pixels: merge is a logical OR, never a subtraction.
    merged = np.maximum(detection.binary_mask, recovered_pixels)

    if apply_final_cleanup:
        final_mask = clean_mask(
            merged,
            threshold=None,  # already binary
            opening_kernel=config.final_opening_kernel,
            closing_kernel=config.final_closing_kernel,
            min_component_size=config.final_min_component_size,
        )
    else:
        final_mask = merged

    average_gap_length = (
        float(np.mean([s.candidate.distance for s in accepted])) if accepted else 0.0
    )

    return GapRecoveryResult(
        detection=detection,
        accepted=accepted,
        rejected=rejected,
        recovered_pixels_mask=recovered_pixels,
        recovered_mask=final_mask,
        average_gap_length=average_gap_length,
    )
