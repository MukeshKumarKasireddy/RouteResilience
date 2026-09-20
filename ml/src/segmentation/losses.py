"""
Loss functions for road segmentation.

Kept separate from the training loop so the loss can be reused in tests,
notebooks, or swapped independently of the training pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss operating on logits.

    Dice loss directly optimizes overlap between prediction and target,
    which helps with the class imbalance typical of road segmentation
    (roads are a small fraction of total pixels).
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy (with logits) + Dice loss.

    A practical default for binary segmentation: BCE provides stable
    per-pixel gradients while Dice pushes for better region overlap,
    which is particularly useful for thin, sparse structures like roads.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def build_loss() -> nn.Module:
    """Factory used by train.py; centralizes loss configuration."""
    return BCEDiceLoss()
