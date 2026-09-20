"""
Inference pipeline for U-Net road segmentation.

Run from the repository root:

    python -m ml.src.segmentation.inference --image-path <path> --model-path <path> --output-dir <dir>

Produces, for a given input image:
    1. a probability mask (PNG, grayscale, values 0-255 scaled from 0-1)
    2. a binary road mask (PNG, grayscale, 0/255)
    3. a metadata JSON describing the prediction, forming the downstream
       contract consumed by Mukesh's geospatial reconstruction pipeline.

Geospatial/CRS metadata is intentionally NOT fabricated here -- it must be
supplied by the upstream processed-imagery pipeline and merged in by the
downstream consumer if needed.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .config import CONFIG, SegmentationConfig
from .dataset import _load_array, _to_rgb  # reuse validated loading logic
from .model import build_model
from .utils import MissingFileError, get_device, get_logger

logger = get_logger(__name__)


def load_model(model_path: str, config: SegmentationConfig = CONFIG, device: Optional[torch.device] = None):
    """Load a trained U-Net model checkpoint for inference."""
    device = device or get_device()
    model_path = Path(model_path)
    if not model_path.exists():
        raise MissingFileError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get("config", {})
    model = build_model(
        in_channels=model_config.get("in_channels", config.in_channels),
        out_channels=model_config.get("out_channels", config.out_channels),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image_path: str, image_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load and preprocess an image for inference.

    Returns the input tensor (1, C, H, W) plus the original (width, height)
    for restoring output masks to the source resolution.
    """
    array = _to_rgb(_load_array(Path(image_path))).astype(np.float32)
    original_height, original_width = array.shape[0], array.shape[1]

    image_pil = Image.fromarray(array.astype(np.uint8)).resize(
        (image_size, image_size), Image.BILINEAR
    )
    array = np.array(image_pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, (original_width, original_height)


@torch.no_grad()
def predict_image(
    image_path: str,
    model_path: str,
    output_dir: str,
    threshold: float = CONFIG.threshold,
    config: SegmentationConfig = CONFIG,
) -> Dict[str, str]:
    """Run inference on a single image and save probability mask, binary
    mask, and metadata JSON to `output_dir`.

    Returns a dict of output file paths: {"probability", "binary", "metadata"}.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, device = load_model(model_path, config)

    input_tensor, (orig_w, orig_h) = preprocess_image(str(image_path), config.image_size)
    input_tensor = input_tensor.to(device)

    logits = model(input_tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W) in [0, 1]

    # Resize probability map back to the original image resolution.
    prob_img = Image.fromarray((probs * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.BILINEAR
    )
    binary_array = (np.array(prob_img).astype(np.float32) / 255.0 > threshold).astype(np.uint8) * 255
    binary_img = Image.fromarray(binary_array)

    stem = image_path.stem
    probability_path = output_dir / f"{stem}_probability.png"
    binary_path = output_dir / f"{stem}_mask.png"
    metadata_path = output_dir / f"{stem}_metadata.json"

    prob_img.save(probability_path)
    binary_img.save(binary_path)

    metadata = {
        "image_name": image_path.name,
        "width": orig_w,
        "height": orig_h,
        "threshold": threshold,
        "model": "U-Net",
        "model_checkpoint": str(model_path),
        "task": "road_segmentation",
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved probability mask -> {probability_path}")
    logger.info(f"Saved binary mask -> {binary_path}")
    logger.info(f"Saved metadata -> {metadata_path}")

    return {
        "probability": str(probability_path),
        "binary": str(binary_path),
        "metadata": str(metadata_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run road segmentation inference on a single image.")
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=str(CONFIG.model_path))
    parser.add_argument("--output-dir", type=str, default=str(CONFIG.predictions_dir))
    parser.add_argument("--threshold", type=float, default=CONFIG.threshold)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    predict_image(
        image_path=args.image_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
