"""
Gap/occlusion detection for cleaned binary road masks (Phase 2).

Road segmentation output is frequently fragmented: clouds, shadows, tree
canopy, overpasses, and model uncertainty can all break an otherwise
continuous road into disconnected components. This module identifies
*candidate* locations where two broken road ends might plausibly belong to
the same real road, without deciding whether to actually reconnect them --
that decision (scoring + conservative acceptance) lives in
`gap_recovery.py`.

Pipeline implemented here:

    cleaned binary mask
        -> validate
        -> connected-component analysis
        -> skeletonization
        -> endpoint detection (per component)
        -> local orientation estimation (per endpoint)
        -> candidate endpoint pairing
        -> distance filtering

The output is a list of `Endpoint` records and a list of `CandidatePair`
records. No pixels are modified by this module -- it only *detects and
describes* candidate gaps.

Ownership: Suryakant (AI/ML Lead) - image-level gap detection only. This
module does not perform geospatial vectorization, CRS handling, or graph
construction.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
except ImportError as exc:  # pragma: no cover - scikit-image is a declared dependency
    raise ImportError(
        "scikit-image is required for gap detection (skeletonize). "
        "Install it via `pip install scikit-image`."
    ) from exc

import cv2


class InvalidMaskError(ValueError):
    """Raised when a mask fails basic validation (shape, dtype, values)."""


@dataclass(frozen=True)
class Endpoint:
    """A single detected skeleton endpoint (candidate broken road tip).

    Attributes:
        x: column (pixel) coordinate.
        y: row (pixel) coordinate.
        component_id: connected-component label this endpoint belongs to.
        orientation: estimated local road direction in radians, pointing
            *outward* from the road (i.e. the direction a recovered
            connection extending from this endpoint would travel in).
        confidence: mean probability-map value in the local neighborhood
            around the endpoint, in [0, 1]. Defaults to 1.0 when no
            probability map is supplied (binary-mask-only evidence).
    """

    x: int
    y: int
    component_id: int
    orientation: float
    confidence: float = 1.0


@dataclass(frozen=True)
class CandidatePair:
    """A candidate connection between two endpoints from different components.

    This record only carries descriptive/geometric information. Whether the
    candidate is actually *accepted* for recovery is decided by
    `gap_recovery.score_candidate` / `gap_recovery.recover_gaps`.
    """

    endpoint_a: Endpoint
    endpoint_b: Endpoint
    distance: float
    orientation_diff: float  # radians, in [0, pi]; 0 = perfectly aligned
    direction_diff: float  # radians, in [0, pi]; angle between the line
    # connecting the two endpoints and each endpoint's own orientation,
    # averaged. 0 = the gap line continues each road's direction exactly.


def validate_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Validate a mask is a 2D binary array and return it as uint8 {0, 1}.

    Raises:
        InvalidMaskError: if the mask is not a 2D array, is empty, or
            contains values outside a binary range after normalization.
    """
    if not isinstance(mask, np.ndarray):
        raise InvalidMaskError(f"Mask must be a numpy array, got {type(mask)}")
    if mask.ndim != 2:
        raise InvalidMaskError(f"Mask must be 2D (H, W), got shape {mask.shape}")
    if mask.size == 0:
        raise InvalidMaskError("Mask must not be empty")

    unique_values = np.unique(mask)
    # Accept {0, 1}, {0, 255}, or already-boolean masks.
    if set(np.unique(np.round(unique_values).astype(int)).tolist()) - {0, 1, 255}:
        raise InvalidMaskError(
            f"Mask must be binary (0/1 or 0/255); found values {unique_values.tolist()}"
        )

    binary = (mask > 0).astype(np.uint8)
    return binary


def detect_connected_components(
    binary_mask: np.ndarray, connectivity: int = 8
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Label connected components in a validated binary mask.

    Returns:
        (num_components, labels, stats) where `labels` is an (H, W) int32
        array of component ids (0 = background) and `stats` is the
        `cv2.connectedComponentsWithStats` stats array indexed by component
        id (stats[0] is the background row).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=connectivity
    )
    num_components = num_labels - 1  # exclude background
    return num_components, labels, stats


def skeletonize_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Thin a binary road mask to a 1-pixel-wide skeleton.

    Uses scikit-image's morphological skeletonization. Returns a uint8
    mask (0/1) of the same shape as the input.
    """
    skeleton = _skimage_skeletonize(binary_mask.astype(bool))
    return skeleton.astype(np.uint8)


def _neighbor_count(skeleton: np.ndarray, y: int, x: int) -> int:
    """Count 8-connected skeleton neighbors of pixel (y, x)."""
    h, w = skeleton.shape
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                count += 1
    return count


def _local_orientation(
    skeleton: np.ndarray, y: int, x: int, radius: int = 7
) -> float:
    """Estimate the outward road direction at a skeleton endpoint.

    Fits a direction from the endpoint towards the "body" of its skeleton
    branch within `radius` pixels (via the mean position of nearby skeleton
    pixels), then returns the *opposite* direction -- i.e. the direction a
    recovered connection extending from this endpoint would travel.

    Falls back to 0.0 radians if no other skeleton pixels are found nearby
    (a fully isolated pixel has no defined orientation).
    """
    h, w = skeleton.shape
    y_min, y_max = max(0, y - radius), min(h, y + radius + 1)
    x_min, x_max = max(0, x - radius), min(w, x + radius + 1)

    local = skeleton[y_min:y_max, x_min:x_max]
    ys, xs = np.nonzero(local)
    if len(ys) <= 1:
        return 0.0

    # Positions relative to the endpoint, in the full-image frame.
    ys_full = ys + y_min
    xs_full = xs + x_min
    mask_self = ~((ys_full == y) & (xs_full == x))
    ys_full, xs_full = ys_full[mask_self], xs_full[mask_self]
    if len(ys_full) == 0:
        return 0.0

    mean_dy = float(np.mean(ys_full - y))
    mean_dx = float(np.mean(xs_full - x))
    if mean_dy == 0.0 and mean_dx == 0.0:
        return 0.0

    # Direction *towards* the body of the road, then flipped to point
    # outward (away from the road), which is the direction a gap
    # connection from this endpoint would extend in.
    inward_angle = np.arctan2(mean_dy, mean_dx)
    outward_angle = inward_angle + np.pi
    return float(np.arctan2(np.sin(outward_angle), np.cos(outward_angle)))


def _mean_confidence(
    probability_map: Optional[np.ndarray], y: int, x: int, radius: int = 3
) -> float:
    """Mean probability-map value in a neighborhood around (y, x).

    Returns 1.0 (full confidence) when no probability map is supplied,
    since binary-only evidence carries no probabilistic distinction.
    """
    if probability_map is None:
        return 1.0
    h, w = probability_map.shape
    y_min, y_max = max(0, y - radius), min(h, y + radius + 1)
    x_min, x_max = max(0, x - radius), min(w, x + radius + 1)
    patch = probability_map[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return 0.0
    return float(np.clip(np.mean(patch), 0.0, 1.0))


def detect_endpoints(
    skeleton: np.ndarray,
    labels: np.ndarray,
    probability_map: Optional[np.ndarray] = None,
    orientation_radius: int = 7,
    confidence_radius: int = 3,
) -> List[Endpoint]:
    """Detect skeleton endpoints for every connected component.

    An endpoint is a skeleton pixel with exactly one 8-connected skeleton
    neighbor. Isolated single-pixel components (no neighbors at all) are
    skipped, since a single pixel has no defined orientation and is more
    likely noise than a genuine road tip.

    Args:
        skeleton: binary skeleton mask (uint8, 0/1), same shape as `labels`.
        labels: connected-component label array from
            `detect_connected_components` (0 = background).
        probability_map: optional float probability mask (0-1), same shape,
            used to estimate per-endpoint confidence.
        orientation_radius: neighborhood radius (pixels) used to estimate
            local road direction at each endpoint.
        confidence_radius: neighborhood radius (pixels) used to average
            probability-map confidence at each endpoint.

    Returns:
        A list of `Endpoint` records. Never raises on noisy/tiny/empty
        skeletons -- simply returns fewer (or zero) endpoints.
    """
    endpoints: List[Endpoint] = []
    ys, xs = np.nonzero(skeleton)

    for y, x in zip(ys.tolist(), xs.tolist()):
        neighbor_count = _neighbor_count(skeleton, y, x)
        if neighbor_count != 1:
            # 0 neighbors: isolated noise pixel, no orientation -> skip.
            # >=2 neighbors: mid-branch or junction pixel, not an endpoint.
            continue

        component_id = int(labels[y, x])
        if component_id == 0:
            continue  # defensive: skeleton pixel with no component label

        orientation = _local_orientation(skeleton, y, x, radius=orientation_radius)
        confidence = _mean_confidence(probability_map, y, x, radius=confidence_radius)

        endpoints.append(
            Endpoint(
                x=x,
                y=y,
                component_id=component_id,
                orientation=orientation,
                confidence=confidence,
            )
        )

    return endpoints


def estimate_endpoint_orientation(
    skeleton: np.ndarray, endpoint: Endpoint, radius: int = 7
) -> float:
    """Public helper to (re-)estimate orientation for a single endpoint.

    Useful for testing or recomputing orientation with a different radius
    without re-running full endpoint detection.
    """
    return _local_orientation(skeleton, endpoint.y, endpoint.x, radius=radius)


def _angle_between(a: float, b: float) -> float:
    """Smallest angle (radians, in [0, pi]) between two angles (undirected)."""
    diff = abs(a - b) % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return diff


def find_candidate_pairs(
    endpoints: List[Endpoint],
    max_distance: float,
) -> List[CandidatePair]:
    """Generate candidate endpoint pairs eligible for gap recovery.

    Pairs are only formed between endpoints belonging to *different*
    connected components (an endpoint cannot be "gapped" to another
    endpoint of the same, already-connected, component) and only when
    within `max_distance` pixels of each other.

    Orientation and direction differences are computed here (as
    descriptive geometry) but *not* used to reject candidates -- rejection
    based on those signals is the responsibility of
    `gap_recovery.filter_candidates` / `gap_recovery.score_candidate`, which
    keeps "what is geometrically possible" separate from "what we choose to
    accept".
    """
    candidates: List[CandidatePair] = []

    for endpoint_a, endpoint_b in combinations(endpoints, 2):
        if endpoint_a.component_id == endpoint_b.component_id:
            continue

        distance = float(
            np.hypot(endpoint_a.x - endpoint_b.x, endpoint_a.y - endpoint_b.y)
        )
        if distance > max_distance or distance <= 0:
            continue

        # Angle of the straight line connecting the two endpoints.
        line_angle = float(
            np.arctan2(endpoint_b.y - endpoint_a.y, endpoint_b.x - endpoint_a.x)
        )

        # Orientation compatibility: endpoints should point roughly *at*
        # each other along the connecting line (i.e. each endpoint's
        # outward orientation should roughly match the line direction,
        # from its own side).
        direction_diff_a = _angle_between(endpoint_a.orientation, line_angle)
        direction_diff_b = _angle_between(
            endpoint_b.orientation, line_angle + np.pi
        )
        direction_diff = float((direction_diff_a + direction_diff_b) / 2.0)

        # Orientation compatibility between the two road directions
        # themselves (are the two road segments roughly collinear?).
        orientation_diff = _angle_between(
            endpoint_a.orientation, endpoint_b.orientation + np.pi
        )

        candidates.append(
            CandidatePair(
                endpoint_a=endpoint_a,
                endpoint_b=endpoint_b,
                distance=distance,
                orientation_diff=orientation_diff,
                direction_diff=direction_diff,
            )
        )

    return candidates


def filter_candidates_by_distance(
    candidates: List[CandidatePair], max_distance: float
) -> List[CandidatePair]:
    """Keep only candidates within `max_distance` pixels.

    `find_candidate_pairs` already applies this filter at generation time;
    this standalone function exists so distance filtering can be reused or
    re-applied independently (e.g. in tests, or when re-filtering an
    existing candidate list with a stricter limit).
    """
    return [c for c in candidates if c.distance <= max_distance]


@dataclass
class GapDetectionResult:
    """Full output of the gap-detection stage, consumed by gap recovery."""

    binary_mask: np.ndarray
    skeleton: np.ndarray
    labels: np.ndarray
    num_components: int
    endpoints: List[Endpoint] = field(default_factory=list)
    candidates: List[CandidatePair] = field(default_factory=list)


def detect_gaps(
    binary_mask: np.ndarray,
    probability_map: Optional[np.ndarray] = None,
    max_distance: float = 60.0,
    orientation_radius: int = 7,
    confidence_radius: int = 3,
    connectivity: int = 8,
) -> GapDetectionResult:
    """Run the full gap-detection pipeline on a cleaned binary road mask.

    This is the single entry point most callers (e.g. `gap_recovery.py`,
    `pipeline.py`) should use; it composes the smaller functions above.

    Args:
        binary_mask: cleaned binary road mask (0/1 or 0/255).
        probability_map: optional float probability mask (0-1), same shape
            as `binary_mask`, used as supporting confidence evidence.
        max_distance: maximum pixel distance between endpoints to even be
            considered a candidate gap (hard geometric cutoff; further,
            stricter acceptance happens in `gap_recovery`).
        orientation_radius: see `detect_endpoints`.
        confidence_radius: see `detect_endpoints`.
        connectivity: 4 or 8-connectivity for component labeling.

    Returns:
        A `GapDetectionResult` describing components, skeleton, endpoints,
        and candidate pairs. Does not modify `binary_mask`.
    """
    validated = validate_binary_mask(binary_mask)
    num_components, labels, _ = detect_connected_components(
        validated, connectivity=connectivity
    )

    if num_components <= 1:
        # Nothing to reconnect: zero or one component means there are no
        # separate fragments to bridge.
        skeleton = skeletonize_mask(validated) if num_components == 1 else np.zeros_like(validated)
        return GapDetectionResult(
            binary_mask=validated,
            skeleton=skeleton,
            labels=labels,
            num_components=num_components,
            endpoints=[],
            candidates=[],
        )

    skeleton = skeletonize_mask(validated)
    # Re-label the skeleton pixels with the (pre-skeletonization) component
    # ids so endpoints retain their original component identity.
    endpoints = detect_endpoints(
        skeleton,
        labels,
        probability_map=probability_map,
        orientation_radius=orientation_radius,
        confidence_radius=confidence_radius,
    )
    candidates = find_candidate_pairs(endpoints, max_distance=max_distance)

    return GapDetectionResult(
        binary_mask=validated,
        skeleton=skeleton,
        labels=labels,
        num_components=num_components,
        endpoints=endpoints,
        candidates=candidates,
    )
