"""
PyTorch Dataset for road segmentation.

Expects a directory layout of the form:

    <root>/
        images/
            tile_0001.png
            tile_0002.png
            ...
        masks/
            tile_0001.png
            tile_0002.png
            ...

Images and masks are matched by filename stem, not by directory order,
so the two folders do not need to be pre-sorted identically.

This module intentionally only depends on Pillow/OpenCV + numpy + torch,
keeping it lightweight and independent of the geospatial stack owned by
the network-reconstruction component of this project. GeoTIFF inputs are
supported on a best-effort basis via rasterio if installed, but PNG/JPEG
is the primary supported format for Part 1.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import CONFIG
from .utils import InvalidImageError, MissingFileError

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _load_array(path: Path) -> np.ndarray:
    """Load an image file into an HWC numpy array.

    Uses Pillow for standard formats. For GeoTIFF files, attempts to use
    rasterio if it is installed; otherwise falls back to Pillow, which
    handles many (but not all) single/multi-band TIFFs.
    """
    if not path.exists():
        raise MissingFileError(f"Expected file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported file format '{suffix}' for {path}. "
            f"Supported formats: {SUPPORTED_EXTENSIONS}"
        )

    if suffix in (".tif", ".tiff"):
        try:
            import rasterio  # type: ignore

            with rasterio.open(path) as src:
                array = src.read()  # (bands, H, W)
                array = np.transpose(array, (1, 2, 0))
                return array
        except ImportError:
            pass  # fall back to Pillow below

    try:
        with Image.open(path) as img:
            return np.array(img)
    except Exception as exc:  # noqa: BLE001 - re-raise as a clear, typed error
        raise InvalidImageError(f"Failed to load image {path}: {exc}") from exc


def _to_rgb(array: np.ndarray) -> np.ndarray:
    """Coerce an arbitrary-channel image array to 3-channel RGB uint8."""
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] > 3:
        # Multi-channel imagery (e.g. multispectral GeoTIFF): keep the
        # first three bands for compatibility with the RGB model input.
        array = array[..., :3]
    elif array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    return array


def _to_mask(array: np.ndarray) -> np.ndarray:
    """Coerce a mask array to single-channel HxW."""
    if array.ndim == 3:
        array = array[..., 0]
    return array


class RoadSegmentationDataset(Dataset):
    """Dataset pairing satellite/processed images with binary road masks.

    Args:
        images_dir: directory containing input images.
        masks_dir: directory containing corresponding binary masks.
        image_size: target (square) size images/masks are resized to.
        augment: whether to apply light train-time augmentation
            (horizontal/vertical flip + 90-degree rotation).
        transform: optional callable override for custom augmentation,
            receiving and returning a (image_np, mask_np) tuple.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_size: Optional[int] = None,
        augment: bool = False,
        transform: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size or CONFIG.image_size
        self.augment = augment
        self.transform = transform

        if not self.images_dir.exists():
            raise MissingFileError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise MissingFileError(f"Masks directory not found: {self.masks_dir}")

        self.pairs: List[Tuple[Path, Path]] = self._match_pairs()
        if not self.pairs:
            raise MissingFileError(
                f"No matching image/mask pairs found between "
                f"{self.images_dir} and {self.masks_dir}"
            )

    def _match_pairs(self) -> List[Tuple[Path, Path]]:
        image_files = {
            p.stem: p
            for p in self.images_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        }
        mask_files = {
            p.stem: p
            for p in self.masks_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        }
        common_stems = sorted(set(image_files) & set(mask_files))
        return [(image_files[stem], mask_files[stem]) for stem in common_stems]

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if np.random.rand() < 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        if np.random.rand() < 0.5:
            k = np.random.choice([1, 2, 3])
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]

        image = _to_rgb(_load_array(image_path)).astype(np.float32)
        mask = _to_mask(_load_array(mask_path)).astype(np.float32)

        if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
            raise InvalidImageError(
                f"Image/mask dimension mismatch for pair "
                f"({image_path.name}, {mask_path.name}): "
                f"image={image.shape[:2]} mask={mask.shape[:2]}"
            )

        image_pil = Image.fromarray(image.astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        mask_pil = Image.fromarray(mask.astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.NEAREST
        )
        image = np.array(image_pil).astype(np.float32)
        mask = np.array(mask_pil).astype(np.float32)

        # Binarize mask (source masks may be 0/255 or 0/1).
        mask = (mask > (mask.max() / 2 if mask.max() > 0 else 0.5)).astype(np.float32)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        elif self.augment:
            image, mask = self._augment(image, mask)

        # Normalize image to [0, 1].
        image = image / 255.0
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return image_tensor, mask_tensor

    def image_name(self, idx: int) -> str:
        """Return the source image filename for a given dataset index."""
        return self.pairs[idx][0].name
