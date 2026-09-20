"""Tests for segmentation metrics: IoU, Dice, precision, recall."""

import torch

from ml.src.segmentation.metrics import (
    compute_all_metrics,
    dice_score,
    iou_score,
    precision_score,
    recall_score,
)


def test_perfect_prediction_scores_near_one():
    targets = torch.zeros(1, 1, 8, 8)
    targets[0, 0, 2:6, 2:6] = 1.0

    # Logits that saturate to the target after sigmoid.
    logits = (targets * 2 - 1) * 20  # -20 or +20

    assert iou_score(logits, targets) > 0.99
    assert dice_score(logits, targets) > 0.99
    assert precision_score(logits, targets) > 0.99
    assert recall_score(logits, targets) > 0.99


def test_completely_wrong_prediction_scores_near_zero():
    targets = torch.zeros(1, 1, 8, 8)
    targets[0, 0, 0:4, 0:4] = 1.0

    preds = torch.zeros(1, 1, 8, 8)
    preds[0, 0, 4:8, 4:8] = 1.0
    logits = (preds * 2 - 1) * 20

    assert iou_score(logits, targets) < 0.01
    assert dice_score(logits, targets) < 0.01


def test_metrics_handle_empty_mask_without_division_by_zero():
    targets = torch.zeros(1, 1, 8, 8)
    logits = torch.full((1, 1, 8, 8), -20.0)  # predicts all-background

    # Should not raise and should not be NaN/inf.
    metrics = compute_all_metrics(logits, targets)
    for name, value in metrics.items():
        assert value == value, f"{name} is NaN"  # NaN check
        assert value != float("inf")


def test_compute_all_metrics_keys():
    targets = torch.randint(0, 2, (2, 1, 16, 16)).float()
    logits = torch.randn(2, 1, 16, 16)
    metrics = compute_all_metrics(logits, targets)
    assert set(metrics.keys()) == {"iou", "dice", "precision", "recall"}


def test_metrics_support_batched_tensors():
    targets = torch.randint(0, 2, (4, 1, 16, 16)).float()
    logits = torch.randn(4, 1, 16, 16)
    metrics = compute_all_metrics(logits, targets)
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
