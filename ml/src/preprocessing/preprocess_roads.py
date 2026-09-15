from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "spacenet_mumbai"

INPUT_DIR = DATA_DIR / "geojson_roads_speed"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "roads"
)


# ============================================================
# Geometry normalization
# ============================================================

def normalize_geometry(geometry):
    """
    Convert supported road geometries into individual LineStrings.

    LineString:
        returned unchanged.

    MultiLineString:
        split into individual LineStrings.

    Other / invalid geometry:
        ignored.
    """

    if geometry is None or geometry.is_empty:
        return []

    if isinstance(geometry, LineString):
        return [geometry]

    if isinstance(geometry, MultiLineString):
        return [
            line
            for line in geometry.geoms
            if not line.is_empty and len(line.coords) >= 2
        ]

    return []


# ============================================================
# Process one GeoJSON file
# ============================================================

def process_file(input_path):
    """Process one SpaceNet road GeoJSON file."""

    roads = gpd.read_file(input_path)

    normalized_rows = []

    for _, row in roads.iterrows():

        geometries = normalize_geometry(row.geometry)

        for geometry in geometries:

            if not geometry.is_valid:
                continue

            new_row = row.copy()
            new_row.geometry = geometry

            normalized_rows.append(new_row)

    if normalized_rows:
        processed_roads = gpd.GeoDataFrame(
            normalized_rows,
            columns=roads.columns,
            crs=roads.crs,
        )
    else:
        # Preserve the original schema even for empty chips.
        processed_roads = gpd.GeoDataFrame(
            columns=roads.columns,
            crs=roads.crs,
        )

    output_path = OUTPUT_DIR / input_path.name

    processed_roads.to_file(
        output_path,
        driver="GeoJSON",
    )

    return len(roads), len(processed_roads)


# ============================================================
# Main processing pipeline
# ============================================================

def main():

    print("========================================")
    print("   RouteResilience Road Preprocessing")
    print("========================================")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_files = sorted(
        INPUT_DIR.glob("*.geojson")
    )

    if not input_files:
        raise FileNotFoundError(
            f"No GeoJSON files found in: {INPUT_DIR}"
        )

    total_input_features = 0
    total_output_features = 0

    for input_path in input_files:

        input_count, output_count = process_file(
            input_path
        )

        total_input_features += input_count
        total_output_features += output_count

        print(
            f"{input_path.name}: "
            f"{input_count} → {output_count} features"
        )

    print("\n========================================")
    print("             SUMMARY")
    print("========================================")

    print(
        f"Input road features:  {total_input_features}"
    )

    print(
        f"Output road features: {total_output_features}"
    )

    print(
        f"Processed files:      {len(input_files)}"
    )

    print(
        f"Output directory:     {OUTPUT_DIR}"
    )

    print("\n===== PREPROCESSING COMPLETE =====")


if __name__ == "__main__":
    main()