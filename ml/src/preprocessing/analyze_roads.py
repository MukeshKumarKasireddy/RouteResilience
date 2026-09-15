from pathlib import Path
from collections import Counter

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "spacenet_mumbai"
ROADS_DIR = DATA_DIR / "geojson_roads_speed"


def analyze_roads():
    road_files = sorted(ROADS_DIR.glob("*.geojson"))

    if not road_files:
        raise FileNotFoundError(
            f"No GeoJSON files found in: {ROADS_DIR}"
        )

    print("========================================")
    print("     RouteResilience Road Analysis")
    print("========================================")

    print(f"Road files found: {len(road_files)}")

    total_features = 0
    empty_files = []
    invalid_geometries = []
    null_geometries = []

    geometry_types = Counter()
    highway_types = Counter()
    surface_types = Counter()

    all_columns = set()

    for road_file in road_files:
        roads = gpd.read_file(road_file)

        chip_id = road_file.stem.split("_chip")[-1]

        print(f"\n--- Chip {chip_id} ---")
        print(f"Road features: {len(roads)}")

        total_features += len(roads)

        if len(roads) == 0:
            empty_files.append(chip_id)

        all_columns.update(roads.columns)

        # Geometry analysis
        for geometry_type in roads.geometry.geom_type:
            geometry_types[geometry_type] += 1

        null_count = roads.geometry.isna().sum()

        if null_count > 0:
            null_geometries.append(
                (chip_id, int(null_count))
            )

        invalid_count = (~roads.geometry.is_valid).sum()

        if invalid_count > 0:
            invalid_geometries.append(
                (chip_id, int(invalid_count))
            )

        # Road attribute analysis
        if "highway" in roads.columns:
            highway_types.update(
                roads["highway"].dropna().astype(str)
            )

        if "surface" in roads.columns:
            surface_types.update(
                roads["surface"].dropna().astype(str)
            )

    print("\n========================================")
    print("           ANALYSIS SUMMARY")
    print("========================================")

    print(f"Total road features: {total_features}")

    print("\nGeometry types:")
    for geometry_type, count in geometry_types.items():
        print(f"  {geometry_type}: {count}")

    print("\nEmpty road files:")
    if empty_files:
        print(f"  {len(empty_files)}")
        print(f"  Chip IDs: {', '.join(empty_files)}")
    else:
        print("  None")

    print("\nNull geometries:")
    if null_geometries:
        for chip_id, count in null_geometries:
            print(f"  Chip {chip_id}: {count}")
    else:
        print("  None")

    print("\nInvalid geometries:")
    if invalid_geometries:
        for chip_id, count in invalid_geometries:
            print(f"  Chip {chip_id}: {count}")
    else:
        print("  None")

    print("\nHighway types:")
    if highway_types:
        for highway, count in highway_types.most_common():
            print(f"  {highway}: {count}")
    else:
        print("  No highway data")

    print("\nSurface types:")
    if surface_types:
        for surface, count in surface_types.most_common():
            print(f"  {surface}: {count}")
    else:
        print("  No surface data")

    print("\nColumns found across dataset:")
    for column in sorted(all_columns):
        print(f"  {column}")

    print("\n===== ROAD ANALYSIS COMPLETE =====")


if __name__ == "__main__":
    analyze_roads()