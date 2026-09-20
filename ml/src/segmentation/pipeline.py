"""
End-to-end road segmentation + gap recovery pipeline (Phase 2).

Integrates the existing Phase 1 segmentation/inference code with the Phase
2 postprocessing (mask cleanup -> gap detection -> conservative gap
recovery), without duplicating any model-loading or inference logic.

Run from the repository root:

    python -m ml.src.segmentation.pipeline --input <path/to/image.png>

Pipeline:

    load image
        -> load model            (segmentation.inference.load_model)
        -> run inference          (segmentation.inference.predict_image)
        -> probability mask
        -> threshold -> binary mask
        -> cleanup                (postprocessing.mask_cleanup.clean_mask)
        -> detect gaps            (postprocessing.gap_detection.detect_gaps)
        -> score candidates       (postprocessing.gap_recovery.filter_candidates)
        -> recover accepted gaps  (postprocessing.gap_recovery.recover_gaps)
        -> final cleanup
        -> save outputs + metadata

Outputs are written under `<output-dir>/<image-stem>/` (see `run_pipeline`
docstring for the exact file list), which extends -- without breaking --
the existing flat `outputs/predictions/<stem>_*.png` convention used by
`segmentation.inference.predict_image` for plain Phase 1 inference.

Ownership: Suryakant (AI/ML Lead) - image-level segmentation + recovery
pipeline only. Downstream geospatial vectorization, CRS assignment, graph
construction, and dashboard rendering are explicitly out of scope here.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from ..postprocessing.gap_recovery import CONFIG as GAP_CONFIG
from ..postprocessing.gap_recovery import GapRecoveryConfig, recover_gaps
from ..postprocessing.mask_cleanup import clean_mask
from .config import CONFIG, SegmentationConfig
from .inference import load_model, preprocess_image
from .utils import get_logger

import torch

logger = get_logger(__name__)


def _save_mask_png(mask: np.ndarray, path: Path) -> None:
    """Save a 0/1 (or 0/255) mask array as an 8-bit grayscale PNG."""
    scaled = (mask.astype(np.float32) * (255 if mask.max() <= 1 else 1)).astype(np.uint8)
    Image.fromarray(scaled).save(path)


def run_pipeline(
    image_path: str,
    model_path: str,
    output_dir: str,
    threshold: float = CONFIG.threshold,
    recovery_enabled: bool = True,
    seg_config: SegmentationConfig = CONFIG,
    gap_config: GapRecoveryConfig = GAP_CONFIG,
) -> Dict[str, str]:
    """Run the full Phase 1 + Phase 2 pipeline on a single image.

    Args:
        image_path: path to the input image.
        model_path: path to a trained U-Net checkpoint (see
            `segmentation.train`).
        output_dir: parent directory; outputs are written to
            `<output_dir>/<image-stem>/`.
        threshold: probability threshold used to binarize the segmentation
            output.
        recovery_enabled: if False, gap detection/recovery is skipped and
            `recovered_mask.png` is simply the cleaned mask -- useful as a
            baseline/ablation and for the "no-recovery" test case.
        seg_config: segmentation configuration (model size, etc).
        gap_config: gap-detection/recovery thresholds and weights.

    Returns:
        A dict of output file paths, keys:
            "original", "probability", "predicted_mask", "cleaned_mask",
            "gaps", "recovered_mask", "metadata".

    Writes, under `<output_dir>/<image-stem>/`:
        original.png        - copy of the input image (for easy comparison)
        probability.png      - raw model probability mask
        predicted_mask.png   - thresholded, pre-cleanup binary mask
        cleaned_mask.png     - after Phase 1 morphological cleanup
        gaps.png             - detected candidate gap connections overlaid
                                on the cleaned mask (diagnostic visualization)
        recovered_mask.png   - final mask after conservative gap recovery
        metadata.json        - see module docstring / docs for schema
    """
    image_path = Path(image_path)
    stem = image_path.stem
    sample_output_dir = Path(output_dir) / stem
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: segmentation inference (reused, not duplicated) ---
    model, device = load_model(model_path, seg_config)
    input_tensor, (orig_w, orig_h) = preprocess_image(str(image_path), seg_config.image_size)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W) in [0, 1]

    prob_img = Image.fromarray((probs * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.Resampling.BILINEAR
    )
    probability_map = np.array(prob_img).astype(np.float32) / 255.0
    predicted_mask = (probability_map > threshold).astype(np.uint8)

    # --- Phase 1: mask cleanup foundation ---
    cleaned_mask = clean_mask(predicted_mask, threshold=None)

    # --- Phase 2: gap detection + conservative recovery ---
    components_before = int(
        np.max(_component_labels(cleaned_mask)) if cleaned_mask.any() else 0
    )

    if recovery_enabled:
        result = recover_gaps(cleaned_mask, probability_map=probability_map, config=gap_config)
        recovered_mask = result.recovered_mask
        accepted = result.accepted
        candidates = result.detection.candidates
        average_gap_length = result.average_gap_length
        gaps_overlay = _render_gap_overlay(cleaned_mask, result)
        components_after = int(
            np.max(_component_labels(recovered_mask)) if recovered_mask.any() else 0
        )
    else:
        recovered_mask = cleaned_mask
        accepted = []
        candidates = []
        average_gap_length = 0.0
        gaps_overlay = (cleaned_mask * 255).astype(np.uint8)
        components_after = components_before

    # --- Save diagnostic outputs ---
    original_out = sample_output_dir / "original.png"
    probability_out = sample_output_dir / "probability.png"
    predicted_mask_out = sample_output_dir / "predicted_mask.png"
    cleaned_mask_out = sample_output_dir / "cleaned_mask.png"
    gaps_out = sample_output_dir / "gaps.png"
    recovered_mask_out = sample_output_dir / "recovered_mask.png"
    metadata_out = sample_output_dir / "metadata.json"

    Image.open(image_path).convert("RGB").save(original_out)
    prob_img.save(probability_out)
    _save_mask_png(predicted_mask, predicted_mask_out)
    _save_mask_png(cleaned_mask, cleaned_mask_out)
    Image.fromarray(gaps_overlay).save(gaps_out)
    _save_mask_png(recovered_mask, recovered_mask_out)

    metadata = {
        "source_image": image_path.name,
        "image_width": orig_w,
        "image_height": orig_h,
        "model": "U-Net",
        "model_checkpoint": str(model_path),
        "threshold": threshold,
        "components_before": components_before,
        "components_after": components_after,
        "candidate_connections": len(candidates),
        "accepted_connections": len(accepted),
        "average_gap_length": average_gap_length,
        "recovery_enabled": recovery_enabled,
    }
    with open(metadata_out, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Pipeline outputs written to {sample_output_dir}")

    return {
        "original": str(original_out),
        "probability": str(probability_out),
        "predicted_mask": str(predicted_mask_out),
        "cleaned_mask": str(cleaned_mask_out),
        "gaps": str(gaps_out),
        "recovered_mask": str(recovered_mask_out),
        "metadata": str(metadata_out),
    }


def _component_labels(mask: np.ndarray) -> np.ndarray:
    """Small local helper avoiding a circular import of cv2 label counting."""
    import cv2

    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labels


def _render_gap_overlay(cleaned_mask: np.ndarray, result) -> np.ndarray:
    """Render an RGB diagnostic image: cleaned mask (white) with accepted
    recovered connections highlighted (red) and rejected candidates shown
    faintly (yellow), for visual debugging of the recovery decision.
    """
    h, w = cleaned_mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[cleaned_mask > 0] = (255, 255, 255)

    import cv2

    for scored in result.rejected:
        point_a = (scored.candidate.endpoint_a.x, scored.candidate.endpoint_a.y)
        point_b = (scored.candidate.endpoint_b.x, scored.candidate.endpoint_b.y)
        cv2.line(overlay, point_a, point_b, color=(120, 120, 0), thickness=1)

    overlay[result.recovered_pixels_mask > 0] = (255, 0, 0)

    return overlay


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full segmentation + gap-recovery pipeline on a single image."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model-path", type=str, default=str(CONFIG.model_path))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CONFIG.predictions_dir),
        help="Parent output directory; a <image-stem>/ subfolder is created inside it.",
    )
    parser.add_argument("--threshold", type=float, default=CONFIG.threshold)
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="Skip gap detection/recovery (Phase 1 behavior only, for comparison).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_pipeline(
        image_path=args.input,
        model_path=args.model_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        recovery_enabled=not args.no_recovery,
    )


if __name__ == "__main__":
    main()
