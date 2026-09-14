# Postprocessing

Owned by: Suryakant (AI/ML Lead) — image-level mask cleanup and (Part 2) occlusion/gap recovery.

## Part 1 (current)

`mask_cleanup.py` implements the cleanup foundation applied to raw model
output before it is handed downstream:

- thresholding a probability mask into a binary mask
- morphological opening (remove small noise specks)
- morphological closing (fill small holes/gaps)
- small connected-component removal

This is deliberately basic. It does **not** attempt to reconnect roads
broken by real occlusion (clouds, shadows, buildings, trees).

## Part 2 (future)

Occlusion/gap *detection* and *recovery* — inferring and reconstructing
road continuity across occluded regions — will be added as a separate
module here without needing to change `mask_cleanup.py`, `inference.py`,
or the dataset/model code.

## Interface

```python
from ml.src.postprocessing.mask_cleanup import clean_mask

cleaned = clean_mask(probability_mask, threshold=0.5)
```

Input: a 2D numpy array (probability mask in [0, 1] or already-binary mask).
Output: a 2D uint8 numpy array with values in {0, 1}.
