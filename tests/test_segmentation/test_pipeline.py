"""Tests for the end-to-end segmentation + gap-recovery pipeline.

Uses a freshly-initialized (untrained) U-Net checkpoint and a small
synthetic image so the full pipeline can be exercised on CPU without any
real trained model or satellite data. These tests validate that the
pipeline *runs end-to-end and produces the expected artifacts* -- they do
not assert anything about segmentation accuracy (untrained weights produce
meaningless predictions, by design).
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from ml.src.segmentation.config import SegmentationConfig
from ml.src.segmentation.model import build_model
from ml.src.segmentation.pipeline import run_pipeline


@pytest.fixture
def tiny_checkpoint(tmp_path):
    """Save an untrained, small U-Net checkpoint compatible with `load_model`."""
    model = build_model(in_channels=3, out_channels=1)
    checkpoint_path = tmp_path / "tiny_unet.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"in_channels": 3, "out_channels": 1, "image_size": 64},
        },
        checkpoint_path,
    )
    return checkpoint_path


@pytest.fixture
def tiny_image(tmp_path):
    """A small synthetic RGB image with two road-like bright segments."""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    image_path = tmp_path / "sample.png"
    Image.fromarray(array).save(image_path)
    return image_path


@pytest.fixture
def tiny_config(tmp_path):
    return SegmentationConfig(
        image_size=64,
        model_dir=tmp_path,
        predictions_dir=tmp_path / "predictions",
    )


def test_pipeline_runs_end_to_end_and_produces_expected_outputs(
    tiny_checkpoint, tiny_image, tmp_path, tiny_config
):
    output_dir = tmp_path / "outputs"
    results = run_pipeline(
        image_path=str(tiny_image),
        model_path=str(tiny_checkpoint),
        output_dir=str(output_dir),
        seg_config=tiny_config,
    )

    expected_keys = {
        "original",
        "probability",
        "predicted_mask",
        "cleaned_mask",
        "gaps",
        "recovered_mask",
        "metadata",
    }
    assert expected_keys <= set(results.keys())

    for key in expected_keys:
        from pathlib import Path

        assert Path(results[key]).exists(), f"missing output file for {key}"


def test_pipeline_metadata_has_expected_schema(
    tiny_checkpoint, tiny_image, tmp_path, tiny_config
):
    output_dir = tmp_path / "outputs"
    results = run_pipeline(
        image_path=str(tiny_image),
        model_path=str(tiny_checkpoint),
        output_dir=str(output_dir),
        seg_config=tiny_config,
    )

    with open(results["metadata"]) as f:
        metadata = json.load(f)

    expected_fields = {
        "source_image",
        "image_width",
        "image_height",
        "model",
        "threshold",
        "components_before",
        "components_after",
        "candidate_connections",
        "accepted_connections",
        "average_gap_length",
        "recovery_enabled",
    }
    assert expected_fields <= set(metadata.keys())
    assert metadata["image_width"] == 64
    assert metadata["image_height"] == 64
    assert metadata["model"] == "U-Net"
    assert metadata["recovery_enabled"] is True
    # Never fabricate CRS/geospatial fields in this component.
    assert "crs" not in metadata
    assert "latitude" not in metadata
    assert "longitude" not in metadata


def test_pipeline_no_recovery_flag_skips_gap_recovery(
    tiny_checkpoint, tiny_image, tmp_path, tiny_config
):
    output_dir = tmp_path / "outputs"
    results = run_pipeline(
        image_path=str(tiny_image),
        model_path=str(tiny_checkpoint),
        output_dir=str(output_dir),
        recovery_enabled=False,
        seg_config=tiny_config,
    )

    with open(results["metadata"]) as f:
        metadata = json.load(f)

    assert metadata["recovery_enabled"] is False
    assert metadata["accepted_connections"] == 0
    assert metadata["candidate_connections"] == 0

    # With recovery disabled, recovered_mask.png must equal cleaned_mask.png.
    cleaned = np.array(Image.open(results["cleaned_mask"]))
    recovered = np.array(Image.open(results["recovered_mask"]))
    assert np.array_equal(cleaned, recovered)


def test_pipeline_output_masks_have_correct_dimensions(
    tiny_checkpoint, tiny_image, tmp_path, tiny_config
):
    output_dir = tmp_path / "outputs"
    results = run_pipeline(
        image_path=str(tiny_image),
        model_path=str(tiny_checkpoint),
        output_dir=str(output_dir),
        seg_config=tiny_config,
    )

    for key in ("predicted_mask", "cleaned_mask", "recovered_mask"):
        img = Image.open(results[key])
        assert img.size == (64, 64)
