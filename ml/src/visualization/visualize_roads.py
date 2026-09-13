from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show


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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualizations"
OUTPUT_PATH = OUTPUT_DIR / "mumbai_chip0_roads_overlay.png"


def create_overlay():
    """Create a satellite-image and road-network overlay."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    roads = gpd.read_file(ROADS_PATH)

    with rasterio.open(IMAGE_PATH) as src:
        fig, ax = plt.subplots(figsize=(10, 10))

        # Display RGB satellite imagery.
        show(src, ax=ax)

        # Plot road network on top of the imagery.
        roads.plot(
            ax=ax,
            linewidth=1.5,
            edgecolor="red",
        )

        ax.set_title("SpaceNet Mumbai — Road Network Overlay")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.tight_layout()
        plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Overlay saved to: {OUTPUT_PATH}")


def main():
    """Run visualization."""

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Satellite image not found: {IMAGE_PATH}")

    if not ROADS_PATH.exists():
        raise FileNotFoundError(f"Road GeoJSON not found: {ROADS_PATH}")

    create_overlay()

    print("Visualization complete.")


if __name__ == "__main__":
    main()