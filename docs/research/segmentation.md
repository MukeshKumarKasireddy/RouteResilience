# Road Segmentation — Research & Design Notes

**Component owner:** Suryakant (AI/ML Lead)
**Scope:** image-level road segmentation, mask cleanup, and occlusion/gap
detection + recovery (Phases 1 and 2 of the CV pipeline)

> Person 1 (Suryakant) recovers road information at the image level.
> The downstream geospatial/network team reconstructs the usable
> geospatial road network. This boundary is not blurred by this component.

This document describes the design decisions behind the road segmentation
(`ml/src/segmentation/`) and gap detection/recovery
(`ml/src/postprocessing/`) implementation.

---

## 1. Problem Definition

Given a satellite or pre-processed aerial image tile, predict, per pixel,
the probability that the pixel belongs to a road. This is framed as **binary
semantic segmentation**: road vs. non-road.

The output of this component feeds a downstream pipeline that reconstructs
a geospatial road network, evaluates disaster resilience/criticality, and
presents results on a dashboard. This component is only responsible for the
image-level extraction step — it does not attempt vectorization, CRS
handling, graph construction, or visualization.

## 2. Why U-Net for the First Implementation

U-Net was selected for Part 1 because it is:

- **Practical** — a well-understood, well-documented architecture with
  predictable training behavior, minimizing time lost to debugging exotic
  failure modes during a time-constrained build.
- **Computationally manageable** — trains and runs inference on CPU-only
  hardware for smoke tests, and scales reasonably on a single GPU for real
  training, without requiring large pretrained backbones.
- **Sufficient for a first working pipeline** — its encoder/decoder with
  skip connections is a strong baseline for thin, elongated structures like
  roads, since skip connections preserve the fine spatial detail that
  aggressive downsampling would otherwise destroy.
- **Simple to extend** — a single plain CNN with no attention blocks or
  ensembling, keeping the codebase easy to reason about and modify.

A more complex architecture (e.g. an ensemble, or starting directly with
DeepLabV3+) was deliberately avoided for Part 1 per the project's ownership
and scope constraints — see Section 9 for how the codebase stays open to
that upgrade later.

## 3. Input Format

- Primary format: PNG/JPEG, 3-channel RGB, arbitrary source resolution.
- Best-effort GeoTIFF support: if `rasterio` is installed, `.tif`/`.tiff`
  files are read via `rasterio`; otherwise the loader falls back to Pillow.
  Multi-band imagery has its first three bands taken as an RGB
  approximation for compatibility with the current model's 3-channel input.
- Images are resized to a configurable square `image_size` (default `256`)
  and normalized to `[0, 1]` before being passed to the model.
- Directory contract: an `images/` and a `masks/` directory (for training),
  matched by filename stem — see `ml/src/segmentation/dataset.py`.

## 4. Output Format

For every prediction, three artifacts are produced (see
`ml/src/segmentation/inference.py`):

1. **Probability mask** — grayscale PNG, pixel values `0-255` representing
   a scaled `0.0-1.0` road probability, at the original input resolution.
2. **Binary road mask** — grayscale PNG, pixel values `{0, 255}`, thresholded
   from the probability mask (default threshold `0.5`, configurable).
3. **Metadata JSON** — describes the prediction (see Section 8, Downstream
   Contract).

## 5. Loss Function

`BCEDiceLoss` (`ml/src/segmentation/losses.py`): an equal-weighted
combination of `BCEWithLogitsLoss` and soft `DiceLoss`.

- BCE provides stable, well-behaved per-pixel gradients.
- Dice loss directly optimizes region overlap, which matters because roads
  typically occupy a small fraction of total pixels — a pure BCE objective
  can be dominated by the (easy) background class.

## 6. Evaluation Metrics

Implemented in `ml/src/segmentation/metrics.py`, all operating on batched
tensors with epsilon smoothing to avoid division-by-zero on empty masks:

- **IoU** (Intersection-over-Union)
- **Dice / F1**
- **Precision**
- **Recall**

`compute_all_metrics(...)` returns all four together and is used in both
the validation loop and the test suite.

## 7. Inference Flow

```
image path
  -> load + coerce to RGB (dataset._load_array / _to_rgb)
  -> resize to model input size, normalize to [0, 1]
  -> UNet forward pass -> logits
  -> sigmoid -> probability map
  -> resize back to original resolution
  -> threshold -> binary mask
  -> save probability PNG, binary PNG, metadata JSON
```

`predict_image(...)` in `inference.py` is the single reusable entry point;
it is used identically by the CLI (`python -m ml.src.segmentation.inference`)
and by the demonstration notebook.

Basic mask cleanup (`ml/src/postprocessing/mask_cleanup.py`) — thresholding,
morphological opening/closing, small-component removal — is applied as a
**separate, optional** step, not baked into `inference.py`, so that
inference output remains the raw model prediction and cleanup remains an
independently testable/replaceable stage.

## 8. Downstream Contract

Every prediction's metadata JSON has the shape:

```json
{
  "image_name": "tile_0001.png",
  "width": 1024,
  "height": 1024,
  "threshold": 0.5,
  "model": "U-Net",
  "model_checkpoint": "ml/models/unet_road_segmentation.pth",
  "task": "road_segmentation"
}
```

Together with the saved probability mask and binary mask files, this forms
the conceptual output: **predicted road probability + predicted binary road
mask + metadata**.

This component does **not** fabricate CRS/georeferencing information. Any
geospatial metadata (coordinate reference system, affine transform, etc.)
must originate from the upstream processed-imagery pipeline and be attached
by the downstream consumer (Mukesh's geospatial reconstruction component)
when it ingests these masks.

## 9. Why Segmentation Produces Broken Roads (Phase 2 motivation)

U-Net (or any per-pixel classifier) predicts road probability independently
per receptive field. Several common conditions cause it to under-predict
road pixels in short stretches, fragmenting an otherwise continuous road
into multiple disconnected mask components:

- **Occlusion**: cloud cover, cloud shadows, and tree canopy overhanging
  the road physically hide the road surface from the sensor.
- **Visual ambiguity**: overpasses, bridges, parking lots, and building
  shadows can locally resemble non-road surfaces or vice versa.
- **Low contrast**: unpaved roads, sun glare, and low-light imagery reduce
  the contrast between road and surrounding terrain.
- **Model uncertainty near the training distribution's edges**: a
  from-scratch, small U-Net trained on limited data (see Limitations) is
  more prone to locally-low-confidence dips than a model with a strong
  pretrained backbone.

A downstream network-reconstruction step needs *connected* road geometry
to build a usable graph; a mask riddled with small gaps produces spurious
disconnected road segments. This motivates Phase 2: not fixing the
segmentation model itself, but conservatively repairing plausible breaks
in its output.

## 10. Why Post-Processing Is Necessary

Two options exist for handling fragmentation: improve the segmentation
model (e.g. more data, a stronger architecture, post-hoc calibration), or
add a downstream heuristic/learned repair step. Phase 2 takes the second
approach for this hackathon timeline because:

- It is **decoupled from model quality** — it works on top of whatever
  segmentation model is deployed (U-Net today, potentially DeepLabV3+
  later) without retraining.
- It is **fast to build and validate** on synthetic data, since gap
  geometry (distance, alignment) is directly inspectable and testable,
  unlike end-to-end model behavior.
- It is **explainable**: every accepted/rejected connection has a
  documented, inspectable score breakdown (Section 13), which matters for
  a system whose output feeds disaster-response decisions downstream.

It does not replace the value of improving the segmentation model itself
over time — see Section 17 (Limitations).

## 11. How Gap Detection Works

Implemented in `ml/src/postprocessing/gap_detection.py`. Given a cleaned
binary mask:

1. **Validate** the mask is 2D and binary (`validate_binary_mask`).
2. **Label connected components** (`detect_connected_components`, via
   `cv2.connectedComponentsWithStats`, 8-connectivity by default). Each
   disconnected road fragment gets its own component id.
3. **Skeletonize** the mask (`skeletonize_mask`, via
   `skimage.morphology.skeletonize`) to a 1-pixel-wide centerline per
   component. This gives a clean geometric basis for endpoint/orientation
   estimation that is insensitive to road width.
4. **Detect endpoints** (`detect_endpoints` — see Section 12).
5. **Pair candidate endpoints** (`find_candidate_pairs` — see Section 13)
   across different components within a configurable `max_gap_distance`.

Nothing in this module modifies mask pixels — it only describes candidate
gaps. This separation (detect vs. decide vs. recover) is deliberate: it
keeps each stage independently testable and means a future change to
scoring/acceptance logic never has to touch the detection geometry code.

## 12. How Endpoints Are Identified

A skeleton pixel is classified as an **endpoint** if it has exactly one
8-connected skeleton neighbor (`_neighbor_count(...) == 1`):

- 0 neighbors → an isolated single pixel (skeleton noise); skipped, since
  it has no defined direction and is more likely noise than a genuine road
  tip.
- 1 neighbor → an endpoint: the tip of a road branch.
- ≥2 neighbors → a mid-branch pixel or a junction; not an endpoint.

For each endpoint, `_local_orientation` estimates the outward road
direction by averaging the relative position of nearby skeleton pixels
within `orientation_radius` (default 7px), then reversing that direction
(since it points *towards* the road body, and a recovered connection would
extend *away* from it, outward). If no other skeleton pixels are nearby
(a fully isolated pixel), orientation defaults to `0.0` radians — a
neutral placeholder that cannot itself pass the orientation-compatibility
gate in Section 13, so it cannot spuriously trigger acceptance.

If a probability map is supplied, `_mean_confidence` also records the mean
prediction confidence in a small neighborhood (`confidence_radius`, default
3px) around the endpoint; this becomes the endpoint's `confidence` field,
used later as scoring evidence.

Edge cases explicitly handled without crashing: isolated pixels, tiny
components, noisy/random skeletons, closed loops (no true endpoints), and
components entirely without endpoints (see `test_gap_detection.py`).

## 13. How Candidate Pairs Are Selected and Scored

**Pairing** (`find_candidate_pairs`, in `gap_detection.py`): every pair of
endpoints from *different* components within `max_gap_distance` pixels
becomes a `CandidatePair`, carrying:

- `distance` — Euclidean distance between the two endpoints.
- `orientation_diff` — angular difference between the two endpoints' own
  local road directions (are the two road segments roughly collinear?).
- `direction_diff` — angular difference between the straight line
  connecting the endpoints and each endpoint's own outward orientation
  (does the gap continue each road's existing direction, or does it cut
  across at an angle?).

Pairing does **not** reject candidates by itself (beyond the hard
`max_gap_distance` cutoff) — it only describes geometry. Acceptance is a
separate decision, made in `gap_recovery.py`.

**Scoring** (`score_candidate`, in `gap_recovery.py`): each of the four
signals is mapped to a `[0, 1]` sub-score via linear interpolation between
a "perfect" and a "worst acceptable" value (`_linear_score`):

| Signal | 1.0 (best) at | 0.0 (worst) at |
|---|---|---|
| `distance_score` | 0 px | `max_gap_distance` |
| `orientation_score` | 0 rad diff | `max_orientation_diff` |
| `direction_score` | 0 rad diff | `max_direction_diff` |
| `confidence_score` | mean endpoint confidence, directly (no interpolation) |

The combined score is a configurable weighted sum:

```
score = w_distance     * distance_score
      + w_orientation   * orientation_score
      + w_direction     * direction_score
      + w_confidence    * confidence_score
```

Default weights (`GapRecoveryConfig`): distance 0.30, orientation 0.30,
direction 0.25, confidence 0.15 — geometry (distance + orientation, 0.60
combined) is weighted more heavily than raw model confidence, since a
geometrically implausible connection should not be rescued by high
confidence at the two endpoints (confidence only reflects "this pixel
looks like road", not "these two fragments are the same road").

## 14. Why Recovery Is Conservative

A candidate is accepted only if it clears **every** hard gate *and* the
combined score exceeds `min_score` (`score_candidate` /
`filter_candidates`):

1. `distance <= max_gap_distance`
2. `orientation_diff <= max_orientation_diff`
3. `direction_diff <= max_direction_diff`
4. `confidence_score >= min_confidence`
5. `score >= min_score`

This two-stage design (independent hard gates, *then* a soft combined
score) is what makes the system conservative: a candidate cannot "buy
back" a bad orientation with a very short distance, or a low confidence
with perfect alignment — every signal must clear its own bar
independently, and even after clearing all four gates the *combined*
score must still be high enough. All five thresholds/weights are
configurable (`GapRecoveryConfig`, overridable via environment variables
such as `GAP_MAX_DISTANCE`, `GAP_MIN_SCORE`) — nothing here is an
unexplained magic number.

Additionally, `filter_candidates` prevents **endpoint reuse**: if two
accepted-scoring candidates share an endpoint, only the higher-scoring one
is kept (the other is demoted to rejected, `rejection_reason
= "endpoint_reused"`). This avoids a single road tip "recovering" into
multiple directions at once, which would not correspond to any real road
topology.

The guiding principle, stated directly: **prefer false negatives over
inventing roads.** A real gap that goes unrecovered can still be corrected
by a human reviewer or a future model iteration; a fabricated road segment
silently corrupts the downstream network reconstruction and resilience
analysis with no easy way to detect it later.

## 15. How Confidence Is Used

When a probability map is available (the raw sigmoid output from Phase 1,
before thresholding), `detect_endpoints` records the mean probability-map
value in a small neighborhood around each endpoint as that endpoint's
`confidence` (Section 12). This is treated purely as **supporting
evidence**, not ground truth:

- It contributes one of four weighted sub-scores (`confidence_score`,
  default weight 0.15 — the lowest of the four), so a confident-looking
  pair can still be rejected on geometry alone.
- It also acts as a hard gate (`min_confidence`): even a well-aligned,
  short-distance candidate is rejected if the model itself was not at
  least moderately confident near the break, since a low-confidence region
  may indicate no road exists there at all (rather than an occluded one).
- If no probability map is supplied, `confidence` defaults to `1.0` for
  every endpoint (i.e. the confidence gate/sub-score becomes a no-op on
  binary-only evidence) — this is a deliberate fallback so the module
  remains usable when only a binary mask is available, not a claim that
  binary-only evidence is actually high-confidence.

The pipeline (`pipeline.py`) always passes the real probability map from
inference into `recover_gaps`, so this fallback only applies to direct,
standalone use of `gap_detection`/`gap_recovery` without a probability map.

Diagnostics preserve the distinction between model prediction (
`probability.png`, `predicted_mask.png`), cleaned mask (`cleaned_mask.png`),
detected gaps (`gaps.png`), and recovered pixels (visible as the red
overlay in `gaps.png` and reflected in `recovered_mask.png`) — see Section
19 (How to Run the Pipeline) and Section 20 (downstream handoff) for the
full output list. Nothing overwrites or conflates these stages.

## 16. Gap Recovery Mechanics

Implemented in `gap_recovery.py`. For each **accepted** candidate:

1. `rasterize_connection` draws a straight line (via `cv2.line`) between
   the two endpoints onto a blank mask of the same shape, with a
   configurable line width (`recovered_line_width`, default 2px).
2. All accepted connections are combined via `np.maximum` (logical OR)
   into a single `recovered_pixels_mask`.
3. This is merged with the original cleaned mask, again via `np.maximum`
   — a logical OR, never a subtraction — so **original road pixels are
   always preserved** by construction; recovery can only add pixels.
4. An optional final cleanup pass (`clean_mask`, same function used in
   Phase 1) removes any small spurious debris introduced by the merge.
   Since morphological opening/closing are idempotent, running this final
   pass on a mask that was already cleaned once (the normal pipeline
   usage) does not erode any pixel that survived the first cleanup pass.

**Straight-line interpolation** was chosen deliberately as the simplest
robust method: within the bounded `max_gap_distance` (default 60px), real
road curvature is well approximated by a straight segment, and a straight
line introduces no free curve parameters that could themselves fabricate
unjustified geometry. A curved/spline interpolation was considered but not
implemented for Phase 2, since it would add complexity without a clear
accuracy benefit at these short gap distances — see Section 17
(Limitations) for when this assumption would break down.

## 17. Limitations

**Phase 1 (segmentation):**

- Trained/demonstrated only on lightweight synthetic sample data
  (`ml/src/segmentation/generate_sample_data.py`) in the absence of a real
  labeled satellite road dataset. **Synthetic-data results do not represent
  real-world road-segmentation performance** and should not be reported as
  such.
- No pretrained backbone / transfer learning; the encoder is trained from
  scratch, which typically needs more labeled data to reach strong
  real-world accuracy than a pretrained-encoder model would.
- Multi-band/multispectral GeoTIFF support is best-effort (first three
  bands only); it is not a full multispectral modeling pipeline.
- Single fixed square input resolution per run (no native multi-scale or
  tiling-and-stitching logic for very large scenes yet).

**Phase 2 (gap detection + recovery):**

- Straight-line recovery cannot represent genuinely curved roads across a
  gap (e.g. a bend fully hidden under cloud cover); it will either reject
  the connection (if the resulting line poorly matches both endpoints'
  orientations) or recover a geometrically-plausible-but-imprecise
  straight segment.
- Endpoint orientation is estimated from a local pixel neighborhood
  (`orientation_radius`, default 7px); very short road stubs, sharp
  curves near the break, or heavy skeleton noise can produce an unreliable
  orientation estimate, which the orientation/direction gates are
  designed to catch and reject — but a wrong-but-plausible-looking
  estimate could in principle pass.
- Recovery only bridges **two** endpoints at a time in a straight line; it
  does not reason about junctions, more than two nearby fragments, or
  road width/topology changes across the gap.
- Confidence gating depends entirely on the quality of the upstream
  probability map; a systematically overconfident or underconfident
  segmentation model will bias which gaps get recovered.
- Default thresholds (`GapRecoveryConfig`) were chosen to be reasonable
  and conservative but are **not** empirically tuned against a labeled
  real-world gap-recovery dataset, since none was available for this
  hackathon timeline; they should be revisited once real satellite/road
  data with known breaks is available.
- Synthetic-data recovery results (see the demonstration notebook) show
  the mechanism working correctly on clean, deterministic test geometry —
  they do not represent real-world recovery accuracy any more than Phase
  1's synthetic segmentation results represent real segmentation accuracy.

## 18. Failure Cases

Documented, expected ways the system can fail *safely* (by rejecting) or
*visibly* (in a way diagnostics reveal), rather than silently producing
wrong output:

- **Two unrelated nearby roads** (e.g. a parallel service road) — rejected
  if their orientations/directions are incompatible; may be incorrectly
  *accepted* if they happen to be short, well-aligned, and confidently
  predicted despite being genuinely different roads. This is the main
  residual risk of any geometry-based heuristic and is why recovery stays
  conservative rather than aggressive.
- **A real gap wider than `max_gap_distance`** — correctly left
  unrecovered (a false negative by design; see Section 14).
- **A junction/intersection mistaken for a broken endpoint** — a T or +
  junction skeleton pixel has ≥2 neighbors and is therefore never
  classified as an endpoint (Section 12), so this should not occur in
  practice; heavily noisy skeletons near junctions could still produce
  spurious endpoint-like pixels, which the distance/orientation/confidence
  gates are the primary defense against.
- **Noisy segmentation output with many tiny components** — `mask_cleanup`
  removing small components before gap detection runs is the primary
  defense; if noise components are large enough to survive cleanup, gap
  detection may generate many low-scoring candidates, all correctly
  rejected, at some added computation cost.
- **No probability map supplied** — confidence defaults to `1.0`
  (Section 15), meaning the confidence gate/sub-score becomes a no-op;
  acceptance then depends entirely on geometry. This is a known, documented
  fallback rather than a hidden failure mode.

## 19. How to Run the Pipeline

```bash
# One-time setup
pip install -r requirements.txt
python -m ml.src.segmentation.generate_sample_data   # only if no real data yet

# Train (Phase 1)
python -m ml.src.segmentation.train

# Run the full Phase 1 + Phase 2 pipeline on one image
python -m ml.src.segmentation.pipeline --input <path/to/image.png>

# Or Phase 1 only (no gap recovery), for comparison/ablation
python -m ml.src.segmentation.pipeline --input <path/to/image.png> --no-recovery

# Tests (CPU only)
pytest tests/test_segmentation/ -v
```

`run_pipeline` (also importable directly from `ml.src.segmentation.pipeline`)
writes, per input image, to `<output-dir>/<image-stem>/`:

```
original.png        - copy of the input image
probability.png      - raw model probability mask (Phase 1)
predicted_mask.png   - thresholded, pre-cleanup binary mask (Phase 1)
cleaned_mask.png     - after Phase 1 morphological cleanup
gaps.png             - diagnostic overlay: cleaned mask (white), accepted
                       recovered connections (red), rejected candidates (faint yellow)
recovered_mask.png   - final mask after conservative gap recovery (Phase 2)
metadata.json        - see Section 20
```

## 20. What Is Handed to the Downstream Geospatial Team

Per processed image, the following files/fields form the Phase 2
downstream contract, consumed by Mukesh's geospatial vectorization and
network-reconstruction stage:

- `recovered_mask.png` — the final image-level road mask (original pixels
  + conservatively recovered connections), the primary input for
  vectorization.
- `cleaned_mask.png` — the pre-recovery mask, retained for comparison/
  auditing or in case a consumer prefers to skip recovered pixels.
- `probability.png` — the raw model confidence map, retained as supporting
  evidence.
- `gaps.png` — gap diagnostics (which connections were made and why),
  useful for debugging or human review, not intended as a vectorization
  input.
- `metadata.json`:

```json
{
  "source_image": "tile_0001.png",
  "image_width": 1024,
  "image_height": 1024,
  "model": "U-Net",
  "model_checkpoint": "ml/models/unet_road_segmentation.pth",
  "threshold": 0.5,
  "components_before": 4,
  "components_after": 2,
  "candidate_connections": 3,
  "accepted_connections": 2,
  "average_gap_length": 34.5,
  "recovery_enabled": true
}
```

This component does **not** perform CRS assignment, geospatial
vectorization, geometry cleanup, graph construction, or any GIS-specific
processing. It does **not** invent latitude/longitude, coordinate
reference systems, or affine transforms — any such metadata must originate
upstream (the processed-imagery pipeline) and be attached by the
downstream consumer. The expected downstream flow, starting from this
component's output, is:

```
recovered_mask.png
    -> vectorization
    -> geometry cleanup
    -> geospatial road network
```

## 21. Future Option: DeepLabV3+

The codebase is structured so DeepLabV3+ (or another architecture) can be
introduced later without a redesign:

- `model.py` exposes a single `build_model(in_channels, out_channels)`
  factory function — swapping the architecture means changing this one
  function's implementation.
- `dataset.py`, `losses.py`, `metrics.py`, `train.py`, and `inference.py`
  only depend on the model's input/output tensor shapes
  (`(B, in_channels, H, W) -> (B, out_channels, H, W)` logits), not on
  U-Net internals.
- `config.py` centralizes hyperparameters, so architecture-specific tuning
  (e.g. different learning rate, backbone choice) can be added without
  scattering new constants across the codebase.

DeepLabV3+ was intentionally **not** the Part 1 architecture — it adds
atrous/dilated convolutions and a heavier decoder that are unnecessary
complexity for a first working hackathon pipeline, per project scope.

## 22. Relationship to Downstream (Mukesh) Pipeline

This component's output — the recovered road mask (Section 20) plus
per-image metadata — is the expected input to Mukesh's geospatial
vectorization and network-reconstruction stage. Ownership boundaries:

| Stage | Owner |
|---|---|
| Road segmentation, probability prediction, binary mask generation | Suryakant |
| Image-level mask cleanup, occlusion/gap detection & recovery | Suryakant |
| Geospatial vectorization, CRS/georeferencing, road geometry, network reconstruction | Mukesh |
| Graph construction, criticality, disaster simulation, resilience metrics | Manogya |
| Dashboard, visualization, product presentation | Paridhi |

No graph construction, NetworkX analysis, dashboard logic, or final
geospatial road-network reconstruction is implemented in this component.
