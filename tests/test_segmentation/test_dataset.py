"""Tests for RoadSegmentationDataset using lightweight synthetic data.

No GPU or real satellite imagery is required.
"""

import numpy as np
import pytest
from PIL import Image

from ml.src.segmentation.dataset import RoadSegmentationDataset
from ml.src.segmentation.utils import MissingFileError


def _write_synthetic_pair(images_dir, masks_dir, name: str, size=(64, 64)):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    mask = (rng.random((size[1], size[0])) > 0.8).astype(np.uint8) * 255

    Image.fromarray(image).save(images_dir / f"{name}.png")
    Image.fromarray(mask).save(masks_dir / f"{name}.png")


@pytest.fixture
def synthetic_dataset_dirs(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    for i in range(4):
        _write_synthetic_pair(images_dir, masks_dir, f"tile_{i:03d}")

    return images_dir, masks_dir


def test_dataset_loads_matching_pairs(synthetic_dataset_dirs):
    images_dir, masks_dir = synthetic_dataset_dirs
    dataset = RoadSegmentationDataset(str(images_dir), str(masks_dir), image_size=32)
    assert len(dataset) == 4


def test_dataset_item_shape_correctness(synthetic_dataset_dirs):
    images_dir, masks_dir = synthetic_dataset_dirs
    image_size = 32
    dataset = RoadSegmentationDataset(str(images_dir), str(masks_dir), image_size=image_size)

    image_tensor, mask_tensor = dataset[0]

    assert image_tensor.shape == (3, image_size, image_size)
    assert mask_tensor.shape == (1, image_size, image_size)
    assert image_tensor.dtype.is_floating_point
    assert mask_tensor.dtype.is_floating_point
    # Mask must be binary after loading.
    unique_values = set(mask_tensor.unique().tolist())
    assert unique_values.issubset({0.0, 1.0})


def test_dataset_missing_images_dir_raises(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    with pytest.raises(MissingFileError):
        RoadSegmentationDataset(str(tmp_path / "does_not_exist"), str(masks_dir))


def test_dataset_no_matching_pairs_raises(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    # No files written -> no matching pairs.
    with pytest.raises(MissingFileError):
        RoadSegmentationDataset(str(images_dir), str(masks_dir))


def test_dataset_image_name_accessor(synthetic_dataset_dirs):
    images_dir, masks_dir = synthetic_dataset_dirs
    dataset = RoadSegmentationDataset(str(images_dir), str(masks_dir), image_size=32)
    name = dataset.image_name(0)
    assert name.endswith(".png")
