"""
Training pipeline for U-Net road segmentation.

Run from the repository root:

    python -m ml.src.segmentation.train

The notebook (ml/notebooks/segmentation_experiment.ipynb) does NOT
duplicate this logic -- it only calls into this module / inference.py.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from .config import CONFIG, SegmentationConfig
from .dataset import RoadSegmentationDataset
from .losses import build_loss
from .metrics import compute_all_metrics
from .model import build_model
from .utils import get_device, get_logger, set_seed

logger = get_logger(__name__)


def build_dataloaders(
    images_dir: str,
    masks_dir: str,
    config: SegmentationConfig,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/validation DataLoaders from a single images+masks directory pair."""
    full_dataset = RoadSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=config.image_size,
        augment=True,
    )

    val_size = max(1, int(len(full_dataset) * config.validation_split))
    train_size = max(1, len(full_dataset) - val_size)

    # If the dataset is too small to split meaningfully, use all data for
    # both train and validation rather than failing outright (useful for
    # smoke tests / synthetic data with very few samples).
    if train_size + val_size > len(full_dataset):
        train_size = len(full_dataset)
        val_size = len(full_dataset)
        train_subset = full_dataset
        val_subset = full_dataset
    else:
        generator = torch.Generator().manual_seed(config.random_seed)
        train_subset, val_subset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device, threshold: float) -> Tuple[float, dict]:
    model.eval()
    running_loss = 0.0
    metric_sums = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
    num_batches = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = loss_fn(logits, masks)
        running_loss += loss.item()

        batch_metrics = compute_all_metrics(logits, masks, threshold=threshold, is_logits=True)
        for key, value in batch_metrics.items():
            metric_sums[key] += value
        num_batches += 1

    num_batches = max(num_batches, 1)
    avg_loss = running_loss / num_batches
    avg_metrics = {key: value / num_batches for key, value in metric_sums.items()}
    return avg_loss, avg_metrics


def train(
    images_dir: Optional[str] = None,
    masks_dir: Optional[str] = None,
    config: SegmentationConfig = CONFIG,
    num_epochs: Optional[int] = None,
) -> Path:
    """Run the full training loop and save the best checkpoint.

    Returns the path to the saved best-model checkpoint.
    """
    set_seed(config.random_seed)
    device = get_device()
    config.ensure_dirs()

    images_dir = images_dir or str(config.samples_dir / "images")
    masks_dir = masks_dir or str(config.samples_dir / "masks")
    epochs = num_epochs or config.num_epochs

    logger.info(f"Using device: {device}")
    logger.info(f"Loading data from images={images_dir} masks={masks_dir}")

    train_loader, val_loader = build_dataloaders(images_dir, masks_dir, config)

    model = build_model(config.in_channels, config.out_channels).to(device)
    loss_fn = build_loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    checkpoint_path = config.model_path

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device, config.threshold)
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_iou={val_metrics['iou']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "config": {
                        "in_channels": config.in_channels,
                        "out_channels": config.out_channels,
                        "image_size": config.image_size,
                    },
                },
                checkpoint_path,
            )
            logger.info(f"Saved new best checkpoint to {checkpoint_path} (val_loss={val_loss:.4f})")

    # Persist a small run summary alongside the model for traceability.
    summary_path = checkpoint_path.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(
            {"best_val_loss": best_val_loss, "epochs_run": epochs, "checkpoint": str(checkpoint_path)},
            f,
            indent=2,
        )

    return checkpoint_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the U-Net road segmentation model.")
    parser.add_argument("--images-dir", type=str, default=None, help="Directory of input images.")
    parser.add_argument("--masks-dir", type=str, default=None, help="Directory of ground-truth masks.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train(images_dir=args.images_dir, masks_dir=args.masks_dir, num_epochs=args.epochs)


if __name__ == "__main__":
    main()
