"""
Evaluation metrics for binary road segmentation.

All functions operate on batched tensors and are safe against
division-by-zero on empty masks (e.g. tiles with no road pixels).
"""

from typing import Dict

import torch

_EPS = 1e-7


def _binarize(logits_or_probs: torch.Tensor, threshold: float, is_logits: bool) -> torch.Tensor:
    probs = torch.sigmoid(logits_or_probs) if is_logits else logits_or_probs
    return (probs > threshold).float()


def iou_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    is_logits: bool = True,
) -> float:
    """Intersection-over-Union for binary masks, averaged over the batch."""
    preds_bin = _binarize(preds, threshold, is_logits)
    targets = targets.float()

    preds_flat = preds_bin.reshape(preds_bin.size(0), -1)
    targets_flat = targets.reshape(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    iou = (intersection + _EPS) / (union + _EPS)
    return iou.mean().item()


def dice_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    is_logits: bool = True,
) -> float:
    """Dice coefficient (F1 over pixels) for binary masks."""
    preds_bin = _binarize(preds, threshold, is_logits)
    targets = targets.float()

    preds_flat = preds_bin.reshape(preds_bin.size(0), -1)
    targets_flat = targets.reshape(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    denom = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2 * intersection + _EPS) / (denom + _EPS)
    return dice.mean().item()


def precision_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    is_logits: bool = True,
) -> float:
    """Precision: of predicted road pixels, how many are actually road."""
    preds_bin = _binarize(preds, threshold, is_logits)
    targets = targets.float()

    preds_flat = preds_bin.reshape(preds_bin.size(0), -1)
    targets_flat = targets.reshape(targets.size(0), -1)

    true_positive = (preds_flat * targets_flat).sum(dim=1)
    predicted_positive = preds_flat.sum(dim=1)
    precision = (true_positive + _EPS) / (predicted_positive + _EPS)
    return precision.mean().item()


def recall_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    is_logits: bool = True,
) -> float:
    """Recall: of actual road pixels, how many were correctly predicted."""
    preds_bin = _binarize(preds, threshold, is_logits)
    targets = targets.float()

    preds_flat = preds_bin.reshape(preds_bin.size(0), -1)
    targets_flat = targets.reshape(targets.size(0), -1)

    true_positive = (preds_flat * targets_flat).sum(dim=1)
    actual_positive = targets_flat.sum(dim=1)
    recall = (true_positive + _EPS) / (actual_positive + _EPS)
    return recall.mean().item()


def compute_all_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    is_logits: bool = True,
) -> Dict[str, float]:
    """Convenience wrapper computing IoU, Dice, precision, and recall together."""
    return {
        "iou": iou_score(preds, targets, threshold, is_logits),
        "dice": dice_score(preds, targets, threshold, is_logits),
        "precision": precision_score(preds, targets, threshold, is_logits),
        "recall": recall_score(preds, targets, threshold, is_logits),
    }
