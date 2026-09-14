# Road Segmentation — Research & Design Notes

**Component owner:** Suryakant (AI/ML Lead)
**Scope:** image-level road segmentation only (Part 1 of the CV pipeline)

> Suryakant owns image-level road extraction and recovery.
> Mukesh owns geospatial/network-level reconstruction.

This document describes the design decisions behind the Part 1 road
segmentation implementation in `ml/src/segmentation/`.

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

## 9. Limitations (Part 1)

- Trained/demonstrated only on lightweight synthetic sample data
  (`ml/src/segmentation/generate_sample_data.py`) in the absence of a real
  labeled satellite road dataset. **Synthetic-data results do not represent
  real-world road-segmentation performance** and should not be reported as
  such.
- No occlusion/gap *recovery* yet — only basic morphological cleanup.
  Reconnecting roads broken by clouds, shadows, buildings, or trees is
  explicitly deferred to Part 2.
- No pretrained backbone / transfer learning; the encoder is trained from
  scratch, which typically needs more labeled data to reach strong
  real-world accuracy than a pretrained-encoder model would.
- Multi-band/multispectral GeoTIFF support is best-effort (first three
  bands only); it is not a full multispectral modeling pipeline.
- Single fixed square input resolution per run (no native multi-scale or
  tiling-and-stitching logic for very large scenes yet).

## 10. Future Option: DeepLabV3+

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

## 11. Relationship to Downstream (Mukesh) Pipeline

This component's output — binary road masks (optionally cleaned) plus
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
