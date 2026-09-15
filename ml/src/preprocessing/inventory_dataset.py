from pathlib import Path
from collections import Counter

import geopandas as gpd
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "spacenet_mumbai"

IMAGE_DIR = DATA_DIR / "PS-RGB"
ROADS_DIR = DATA_DIR / "geojson_roads_speed"


def find_files():
    """Find satellite images and road GeoJSON files."""

    image_files = sorted(IMAGE_DIR.glob("*.tif"))
    road_files = sorted(ROADS_DIR.glob("*.geojson"))

    return image_files, road_files


def extract_chip_id(filename):
    """Extract the chip ID from a SpaceNet filename."""

    marker = "_chip"

    if marker not in filename:
        return None

    return filename.rsplit(marker, 1)[1].split(".")[0]


def inspect_images(image_files):
    """Inspect satellite image files."""

    print("\n===== SATELLITE IMAGES =====")
    print(f"Total TIFF files: {len(image_files)}")

    if not image_files:
        print("No TIFF files found.")
        return

    crs_counter = Counter()
    band_counter = Counter()
    size_counter = Counter()

    for image_path in image_files:
        try:
            with rasterio.open(image_path) as src:
                crs_counter[str(src.crs)] += 1
                band_counter[src.count] += 1
                size_counter[(src.width, src.height)] += 1

        except Exception as exc:
            print(f"ERROR: {image_path.name}")
            print(f"       {exc}")

    print(f"CRS distribution: {dict(crs_counter)}")
    print(f"Band distribution: {dict(band_counter)}")
    print(f"Image-size distribution: {dict(size_counter)}")


def inspect_roads(road_files):
    """Inspect road GeoJSON files."""

    print("\n===== ROAD GEOJSON FILES =====")
    print(f"Total GeoJSON files: {len(road_files)}")

    if not road_files:
        print("No GeoJSON files found.")
        return

    crs_counter = Counter()
    geometry_counter = Counter()
    feature_counts = []

    for road_path in road_files:
        try:
            roads = gpd.read_file(road_path)

            crs_counter[str(roads.crs)] += 1

            for geometry_type in roads.geometry.geom_type:
                geometry_counter[geometry_type] += 1

            feature_counts.append(len(roads))

        except Exception as exc:
            print(f"ERROR: {road_path.name}")
            print(f"       {exc}")

    print(f"CRS distribution: {dict(crs_counter)}")
    print(f"Geometry distribution: {dict(geometry_counter)}")

    if feature_counts:
        print(f"Minimum road features per file: {min(feature_counts)}")
        print(f"Maximum road features per file: {max(feature_counts)}")
        print(
            "Average road features per file: "
            f"{sum(feature_counts) / len(feature_counts):.2f}"
        )


def check_matching_files(image_files, road_files):
    """Check whether image and GeoJSON chip IDs match."""

    print("\n===== IMAGE / ROAD MATCHING =====")

    image_chips = {
        extract_chip_id(path.name)
        for path in image_files
    }

    road_chips = {
        extract_chip_id(path.name)
        for path in road_files
    }

    image_chips.discard(None)
    road_chips.discard(None)

    print(f"Image chip IDs found: {len(image_chips)}")
    print(f"Road chip IDs found: {len(road_chips)}")

    missing_roads = sorted(image_chips - road_chips)
    missing_images = sorted(road_chips - image_chips)

    if missing_roads:
        print(
            "Images without matching road files: "
            f"{len(missing_roads)}"
        )
        print(f"Missing road chip IDs: {missing_roads}")
    else:
        print("Every discovered image has a matching road file.")

    if missing_images:
        print(
            "Road files without matching images: "
            f"{len(missing_images)}"
        )
        print(f"Missing image chip IDs: {missing_images}")
    else:
        print("Every discovered road file has a matching image.")


def main():
    """Build a dataset inventory."""

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    if not ROADS_DIR.exists():
        raise FileNotFoundError(
            f"Road directory not found: {ROADS_DIR}"
        )

    print("========================================")
    print("   RouteResilience Dataset Inventory")
    print("========================================")
    print(f"Dataset directory: {DATA_DIR}")

    image_files, road_files = find_files()

    inspect_images(image_files)
    inspect_roads(road_files)
    check_matching_files(image_files, road_files)

    print("\n===== INVENTORY COMPLETE =====")


if __name__ == "__main__":
    main()