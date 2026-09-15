from pathlib import Path
from collections import Counter

import geopandas as gpd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "raw"
    / "spacenet_mumbai"
)

RAW_DIR = DATA_DIR / "geojson_roads_speed"

PROCESSED_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "roads"
)


# ============================================================
# Validate one file
# ============================================================

def validate_file(raw_path, processed_path):
    """Validate one raw/processed road-data pair."""

    raw = gpd.read_file(raw_path)
    processed = gpd.read_file(processed_path)

    errors = []

    raw_count = len(raw)
    processed_count = len(processed)

    # --------------------------------------------------------
    # CRS validation
    # --------------------------------------------------------

    if raw.crs != processed.crs:
        errors.append(
            f"CRS changed: {raw.crs} -> {processed.crs}"
        )

    # --------------------------------------------------------
    # Geometry validation
    # --------------------------------------------------------

    null_count = int(
        processed.geometry.isna().sum()
    )

    if null_count > 0:
        errors.append(
            f"Processed data contains "
            f"{null_count} null geometries"
        )

    invalid_count = int(
        (~processed.geometry.is_valid).sum()
    )

    if invalid_count > 0:
        errors.append(
            f"Processed data contains "
            f"{invalid_count} invalid geometries"
        )

    # --------------------------------------------------------
    # Geometry type validation
    # --------------------------------------------------------

    geometry_types = set(
        processed.geometry.geom_type.dropna()
    )

    unexpected_types = (
        geometry_types - {"LineString"}
    )

    if unexpected_types:
        errors.append(
            f"Unexpected geometry types: "
            f"{sorted(unexpected_types)}"
        )

    # --------------------------------------------------------
    # Required attribute validation
    # --------------------------------------------------------
    #
    # Empty GeoJSON files may not retain their original
    # feature-property columns because they contain zero
    # features. This is valid for our dataset.
    #
    # Therefore, required attributes are checked only when
    # the processed file contains actual road features.
    # --------------------------------------------------------

    required_columns = {
        "highway",
        "surface",
        "lanes",
        "bridge",
        "osm_id",
        "inferred_speed_mph",
        "inferred_speed_mps",
        "geometry",
    }

    if processed_count > 0:

        missing_columns = (
            required_columns
            - set(processed.columns)
        )

        if missing_columns:
            errors.append(
                f"Missing columns: "
                f"{sorted(missing_columns)}"
            )

    # --------------------------------------------------------
    # Geometry count
    # --------------------------------------------------------

    geometry_count = Counter(
        processed.geometry.geom_type.dropna()
    )

    return {
        "raw_count": raw_count,
        "processed_count": processed_count,
        "geometry_types": geometry_types,
        "geometry_count": geometry_count,
        "errors": errors,
    }


# ============================================================
# Main validation
# ============================================================

def main():

    print("========================================")
    print(
        " RouteResilience Preprocessing Validation"
    )
    print("========================================")

    # --------------------------------------------------------
    # Directory checks
    # --------------------------------------------------------

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw directory not found: {RAW_DIR}"
        )

    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(
            f"Processed directory not found: "
            f"{PROCESSED_DIR}"
        )

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    raw_files = sorted(
        RAW_DIR.glob("*.geojson")
    )

    processed_files = sorted(
        PROCESSED_DIR.glob("*.geojson")
    )

    print(f"Raw files:       {len(raw_files)}")
    print(f"Processed files: {len(processed_files)}")

    # --------------------------------------------------------
    # File matching
    # --------------------------------------------------------

    raw_names = {
        file.name
        for file in raw_files
    }

    processed_names = {
        file.name
        for file in processed_files
    }

    missing_processed = (
        raw_names - processed_names
    )

    unexpected_processed = (
        processed_names - raw_names
    )

    validation_failed = False

    if missing_processed:

        validation_failed = True

        print("\nERROR: Missing processed files:")

        for name in sorted(missing_processed):
            print(f"  {name}")

    if unexpected_processed:

        validation_failed = True

        print("\nERROR: Unexpected processed files:")

        for name in sorted(unexpected_processed):
            print(f"  {name}")

    # --------------------------------------------------------
    # Validate every file
    # --------------------------------------------------------

    total_raw = 0
    total_processed = 0

    total_feature_increase = 0

    geometry_counter = Counter()

    for raw_path in raw_files:

        processed_path = (
            PROCESSED_DIR
            / raw_path.name
        )

        if not processed_path.exists():
            validation_failed = True
            continue

        result = validate_file(
            raw_path,
            processed_path,
        )

        total_raw += result["raw_count"]
        total_processed += result[
            "processed_count"
        ]

        feature_increase = (
            result["processed_count"]
            - result["raw_count"]
        )

        total_feature_increase += (
            feature_increase
        )

        geometry_counter.update(
            result["geometry_count"]
        )

        if result["errors"]:

            validation_failed = True

            print(
                f"\nFAILED: {raw_path.name}"
            )

            for error in result["errors"]:
                print(f"  - {error}")

        else:

            print(
                f"PASS: {raw_path.name} "
                f"({result['raw_count']} -> "
                f"{result['processed_count']})"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n========================================")
    print("             VALIDATION SUMMARY")
    print("========================================")

    print(
        f"Total raw features:       {total_raw}"
    )

    print(
        f"Total processed features: "
        f"{total_processed}"
    )

    print(
        f"Feature increase:         "
        f"{total_feature_increase}"
    )

    print("\nProcessed geometry types:")

    if geometry_counter:

        for geometry_type, count in (
            geometry_counter.most_common()
        ):
            print(
                f"  {geometry_type}: {count}"
            )

    else:
        print("  None")

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    if validation_failed:

        print("\n===== VALIDATION FAILED =====")

        raise SystemExit(1)

    print("\n===== VALIDATION PASSED =====")

    print(
        "Raw and processed road datasets are "
        "consistent and ready for downstream use."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()