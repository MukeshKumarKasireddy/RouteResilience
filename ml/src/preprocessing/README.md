# Preprocessing

This directory is reserved for upstream image preprocessing steps (e.g.
tiling large satellite scenes, cloud/shadow filtering, band selection,
CRS-aware reprojection prior to model input).

For Part 1, no preprocessing beyond what is built directly into
`ml/src/segmentation/dataset.py` (resizing, normalization, format
handling) has been implemented. This README exists as a placeholder so
the intended structure is visible from the start; preprocessing logic
should be added here as the project matures, without altering the
dataset/model/training contracts already established in
`ml/src/segmentation/`.

Ownership: Suryakant (AI/ML Lead) for image-level preprocessing feeding
into segmentation. Any CRS/georeferencing preprocessing that feeds the
geospatial reconstruction pipeline belongs to Mukesh's component instead.
