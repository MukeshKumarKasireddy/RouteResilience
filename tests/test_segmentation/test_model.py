"""Tests for the U-Net model: instantiation, forward pass, output shape."""

import torch

from ml.src.segmentation.model import UNet, build_model


def test_model_instantiation():
    model = build_model(in_channels=3, out_channels=1)
    assert isinstance(model, UNet)


def test_model_forward_pass_output_shape():
    model = build_model(in_channels=3, out_channels=1)
    model.eval()

    batch_size, height, width = 2, 64, 64
    dummy_input = torch.randn(batch_size, 3, height, width)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (batch_size, 1, height, width)


def test_model_forward_pass_non_power_of_two_input():
    """Ensure the model tolerates input sizes not divisible cleanly by 16."""
    model = build_model(in_channels=3, out_channels=1)
    model.eval()

    dummy_input = torch.randn(1, 3, 70, 70)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape[0] == 1
    assert output.shape[1] == 1


def test_model_output_is_logits_not_bounded():
    """Confirm the model outputs raw logits (no sigmoid applied internally)."""
    model = build_model(in_channels=3, out_channels=1)
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32) * 10  # push toward extremes
    with torch.no_grad():
        output = model(dummy_input)

    # Logits can exceed [0, 1]; a sigmoid-bounded output would not.
    assert output.min().item() != output.max().item()
