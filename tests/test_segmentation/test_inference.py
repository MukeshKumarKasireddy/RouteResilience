"""Tests for the inference pipeline using a synthetic image and an
untrained (randomly initialized) checkpoint -- sufficient to verify the
pipeline runs end-to-end and produces the expected output artifacts,
without asserting anything about real segmentation quality.
"""

import json

import numpy as np
import torch
from PIL import Image

from ml.src.segmentation.inference import predict_image
from ml.src.segmentation.model import build_model


def _make_synthetic_checkpoint(path, in_channels=3, out_channels=1):
    model = build_model(in_channels=in_channels, out_channels=out_channels)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "val_loss": 0.0,
            "val_metrics": {},
            "config": {"in_channels": in_channels, "out_channels": out_channels, "image_size": 64},
        },
        path,
    )


def _make_synthetic_image(path, size=(48, 48)):
    rng = np.random.default_rng(1)
    array = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_predict_image_produces_expected_outputs(tmp_path):
    image_path = tmp_path / "sample.png"
    model_path = tmp_path / "model.pth"
    output_dir = tmp_path / "outputs"

    _make_synthetic_image(image_path)
    _make_synthetic_checkpoint(model_path)

    result = predict_image(
        image_path=str(image_path),
        model_path=str(model_path),
        output_dir=str(output_dir),
        threshold=0.5,
    )

    assert "probability" in result and "binary" in result and "metadata" in result

    prob_img = Image.open(result["probability"])
    binary_img = Image.open(result["binary"])
    assert prob_img.size == (48, 48)
    assert binary_img.size == (48, 48)

    binary_array = np.array(binary_img)
    unique_values = set(np.unique(binary_array).tolist())
    assert unique_values.issubset({0, 255})


def test_predict_image_metadata_contract(tmp_path):
    image_path = tmp_path / "sample2.png"
    model_path = tmp_path / "model2.pth"
    output_dir = tmp_path / "outputs2"

    _make_synthetic_image(image_path, size=(32, 40))
    _make_synthetic_checkpoint(model_path)

    result = predict_image(
        image_path=str(image_path),
        model_path=str(model_path),
        output_dir=str(output_dir),
        threshold=0.42,
    )

    with open(result["metadata"]) as f:
        metadata = json.load(f)

    required_keys = {
        "image_name",
        "width",
        "height",
        "threshold",
        "model",
        "model_checkpoint",
        "task",
    }
    assert required_keys.issubset(metadata.keys())
    assert metadata["width"] == 32
    assert metadata["height"] == 40
    assert metadata["threshold"] == 0.42
    assert metadata["model"] == "U-Net"
    assert metadata["task"] == "road_segmentation"
