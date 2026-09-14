"""
Utility script to generate lightweight synthetic image/mask pairs under
``ml/data/samples/``.

This is NOT a substitute for real satellite imagery. It exists so the
full pipeline (dataset -> training -> inference -> notebook -> tests)
can be executed and demonstrated end-to-end even when no real dataset
has been added to the repository yet.

Run from the repository root:

    python -m ml.src.segmentation.generate_sample_data
"""

import argparse

import numpy as np
from PIL import Image, ImageDraw

from .config import CONFIG


def _make_synthetic_pair(size: int, seed: int) -> "tuple[Image.Image, Image.Image]":
    """Create one synthetic (image, mask) pair with a few road-like lines.

    The "image" is random terrain-like noise with a handful of straight
    light-colored strokes drawn on top (standing in for roads). The mask
    marks exactly those stroke pixels. This is only useful for exercising
    the pipeline mechanically -- it does not represent real road imagery.
    """
    rng = np.random.default_rng(seed)

    # Terrain-like background noise.
    background = rng.integers(40, 120, size=(size, size, 3), dtype=np.uint8)
    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)

    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)

    num_roads = rng.integers(2, 5)
    for _ in range(num_roads):
        x1, y1 = rng.integers(0, size, size=2)
        x2, y2 = rng.integers(0, size, size=2)
        width = int(rng.integers(2, 5))
        road_color = tuple(int(c) for c in rng.integers(180, 220, size=3))
        draw.line([(x1, y1), (x2, y2)], fill=road_color, width=width)
        mask_draw.line([(x1, y1), (x2, y2)], fill=255, width=width)

    return image, mask


def generate_samples(num_samples: int = 12, size: int = 256) -> None:
    """Generate `num_samples` synthetic image/mask pairs into ml/data/samples/."""
    images_dir = CONFIG.samples_dir / "images"
    masks_dir = CONFIG.samples_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        image, mask = _make_synthetic_pair(size=size, seed=i)
        name = f"synthetic_{i:03d}.png"
        image.save(images_dir / name)
        mask.save(masks_dir / name)

    print(f"Wrote {num_samples} synthetic image/mask pairs to {CONFIG.samples_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic sample data.")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_samples(num_samples=args.num_samples, size=args.size)


if __name__ == "__main__":
    main()
