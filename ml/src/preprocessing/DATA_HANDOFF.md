# RouteResilience — Geospatial Data Handoff

## 1. Purpose

This document describes the prepared SpaceNet Mumbai dataset being handed
over from the Geospatial Data Engineering stage to the AI/ML stage.

The purpose of this stage is to provide Suryakant with:

- matched satellite image chips
- cleaned and validated road-label geometries
- consistent file naming
- validated geospatial metadata
- a clear separation between raw source data and processed labels

The data is prepared for downstream road segmentation and occlusion-recovery
model development.

---

## 2. Dataset

### Source

SpaceNet 5 — Mumbai AOI 8 road dataset.

### Area

Mumbai, India.

### Dataset used

A local subset of the SpaceNet Mumbai training dataset containing:

- 206 satellite image chips
- 206 corresponding road-label GeoJSON files
- 206 matched image/label pairs

Therefore, the dataset contains:

- 206 `.tif` satellite images
- 206 `.geojson` road-label files
- 412 raw dataset files in total

This exceeds the requested minimum of 200 matched image/label pairs.

---

## 3. Raw Data Structure

The original SpaceNet data is stored locally under:

```text
ml/
└── data/
    └── raw/
        └── spacenet_mumbai/
            ├── PS-RGB/
            │   ├── *.tif
            │   └── ...
            │
            └── geojson_roads_speed/
                ├── *.geojson
                └── ...