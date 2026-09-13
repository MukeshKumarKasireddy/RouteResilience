from pathlib import Path

import geopandas as gpd
import rasterio


# Project root:
# RouteResilience/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "raw"
    / "spacenet_mumbai"
    / "chip0"
    / "SN5_roads_train_AOI_8_Mumbai_PS-RGB_chip0.tif"
)

ROADS_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "raw"
    / "spacenet_mumbai"
    / "chip0"
    / "SN5_roads_train_AOI_8_Mumbai_geojson_roads_speed_chip0.geojson"
)


def inspect_image():
    """Inspect the satellite GeoTIFF metadata."""

    print("\n===== SATELLITE IMAGE =====")

    with rasterio.open(IMAGE_PATH) as src:
        print(f"File: {IMAGE_PATH.name}")
        print(f"Width: {src.width}")
        print(f"Height: {src.height}")
        print(f"Bands: {src.count}")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Resolution: {src.res}")
        print(f"Data type: {src.dtypes}")


def inspect_roads():
    """Inspect the GeoJSON road network."""

    print("\n===== ROAD NETWORK =====")

    roads = gpd.read_file(ROADS_PATH)

    print(f"File: {ROADS_PATH.name}")
    print(f"Number of road features: {len(roads)}")
    print(f"CRS: {roads.crs}")
    print(f"Geometry types: {roads.geometry.geom_type.unique()}")
    print(f"Columns: {list(roads.columns)}")
    print(f"Bounds: {roads.total_bounds}")


def main():
    """Run all data inspection checks."""

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Satellite image not found: {IMAGE_PATH}")

    if not ROADS_PATH.exists():
        raise FileNotFoundError(f"Road GeoJSON not found: {ROADS_PATH}")

    inspect_image()
    inspect_roads()

    print("\n===== INSPECTION COMPLETE =====")


if __name__ == "__main__":
    main()