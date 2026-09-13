from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
from shapely.ops import split, unary_union


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_ROADS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "roads"
)


# ============================================================
# Configuration
# ============================================================

# Mumbai is located in UTM Zone 43N.
# This CRS uses metres instead of degrees.
METRIC_CRS = "EPSG:32643"

# Very small tolerance used when deciding whether two
# geometries/intersections are effectively at the same location.
SNAP_TOLERANCE_M = 1.0


# ============================================================
# Node helpers
# ============================================================

def point_key(point):
    """
    Convert a point into a stable graph-node identifier.

    Coordinates are rounded to millimetres after reprojection.
    """

    return (
        round(point.x, 3),
        round(point.y, 3),
    )


# ============================================================
# Attribute helpers
# ============================================================

ROAD_ATTRIBUTES = [
    "highway",
    "surface",
    "lanes",
    "bridge",
    "osm_id",
    "inferred_speed_mph",
    "inferred_speed_mps",
    "partialDec",
    "truncated",
]


def extract_attributes(row):
    """
    Extract useful road attributes from a GeoDataFrame row.
    """

    attributes = {}

    for column in ROAD_ATTRIBUTES:

        if column in row.index:
            attributes[column] = row[column]

    return attributes


# ============================================================
# Intersection detection
# ============================================================

def get_intersection_points(roads):
    """
    Find point intersections between road LineStrings.

    Returns a list of Point geometries.

    We intentionally ignore overlaps for now because overlapping
    roads require more careful handling than simple crossings.
    """

    intersection_points = []

    geometries = list(roads.geometry)

    for i in range(len(geometries)):

        geometry_a = geometries[i]

        if geometry_a is None or geometry_a.is_empty:
            continue

        for j in range(i + 1, len(geometries)):

            geometry_b = geometries[j]

            if geometry_b is None or geometry_b.is_empty:
                continue

            if not geometry_a.intersects(geometry_b):
                continue

            intersection = geometry_a.intersection(
                geometry_b
            )

            if intersection.is_empty:
                continue

            if intersection.geom_type == "Point":

                intersection_points.append(
                    intersection
                )

            elif intersection.geom_type == "MultiPoint":

                intersection_points.extend(
                    list(intersection.geoms)
                )

            elif intersection.geom_type == "GeometryCollection":

                for geometry in intersection.geoms:

                    if geometry.geom_type == "Point":

                        intersection_points.append(
                            geometry
                        )

    return intersection_points


# ============================================================
# Split road at intersections
# ============================================================

def split_line_at_points(line, points):
    """
    Split a LineString at intersection points.

    Only points that lie on the line are used.
    """

    if line is None or line.is_empty:
        return []

    if line.geom_type != "LineString":
        return []

    if not points:
        return [line]

    valid_points = []

    for point in points:

        if point.distance(line) <= SNAP_TOLERANCE_M:

            valid_points.append(point)

    if not valid_points:
        return [line]

    splitter = unary_union(valid_points)

    try:

        result = split(
            line,
            splitter
        )

        return [
            geometry
            for geometry in result.geoms
            if (
                geometry.geom_type == "LineString"
                and geometry.length > 0
            )
        ]

    except Exception:

        # If splitting fails for an individual road,
        # retain the original geometry rather than losing data.
        return [line]


# ============================================================
# Prepare roads
# ============================================================

def reconstruct_roads(roads):
    """
    Reconstruct road geometries by splitting roads at
    detected intersections.
    """

    intersection_points = get_intersection_points(
        roads
    )

    reconstructed = []

    for _, row in roads.iterrows():

        geometry = row.geometry

        if geometry is None or geometry.is_empty:
            continue

        if geometry.geom_type != "LineString":
            continue

        pieces = split_line_at_points(
            geometry,
            intersection_points
        )

        attributes = extract_attributes(row)

        for piece in pieces:

            reconstructed.append(
                {
                    "geometry": piece,
                    **attributes,
                }
            )

    if not reconstructed:

        return gpd.GeoDataFrame(
            columns=[
                "geometry",
                *ROAD_ATTRIBUTES,
            ],
            geometry="geometry",
            crs=roads.crs,
        )

    return gpd.GeoDataFrame(
        reconstructed,
        geometry="geometry",
        crs=roads.crs,
    )


# ============================================================
# Graph construction
# ============================================================

def build_graph(roads):
    """
    Build a NetworkX graph from reconstructed road segments.
    """

    graph = nx.Graph()

    for road_id, (_, row) in enumerate(
        roads.iterrows()
    ):

        geometry = row.geometry

        if geometry is None or geometry.is_empty:
            continue

        if geometry.geom_type != "LineString":
            continue

        coordinates = list(
            geometry.coords
        )

        if len(coordinates) < 2:
            continue

        start_point = Point(
            coordinates[0]
        )

        end_point = Point(
            coordinates[-1]
        )

        start_node = point_key(
            start_point
        )

        end_node = point_key(
            end_point
        )

        if start_node == end_node:
            continue

        attributes = extract_attributes(
            row
        )

        attributes.update(
            {
                "road_id": road_id,
                "length_m": float(
                    geometry.length
                ),
                "geometry": geometry,
            }
        )

        graph.add_node(
            start_node,
            x=start_node[0],
            y=start_node[1],
        )

        graph.add_node(
            end_node,
            x=end_node[0],
            y=end_node[1],
        )

        graph.add_edge(
            start_node,
            end_node,
            **attributes,
        )

    return graph


# ============================================================
# Graph analysis
# ============================================================

def analyze_graph(graph):

    print(
        f"Nodes: {graph.number_of_nodes()}"
    )

    print(
        f"Edges: {graph.number_of_edges()}"
    )

    if graph.number_of_edges() == 0:
        return

    total_length = sum(
        data.get("length_m", 0.0)
        for _, _, data in graph.edges(
            data=True
        )
    )

    components = list(
        nx.connected_components(graph)
    )

    print(
        f"Total road length: "
        f"{total_length:.2f} m"
    )

    print(
        f"Connected components: "
        f"{len(components)}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("========================================")
    print(" RouteResilience Network Reconstruction")
    print("========================================")

    if not PROCESSED_ROADS_DIR.exists():

        raise FileNotFoundError(
            "Processed roads directory not found:\n"
            f"{PROCESSED_ROADS_DIR}"
        )

    road_files = sorted(
        PROCESSED_ROADS_DIR.glob(
            "*.geojson"
        )
    )

    if not road_files:

        raise FileNotFoundError(
            "No processed GeoJSON files found."
        )

    print(
        f"Processed road files: "
        f"{len(road_files)}"
    )

    total_input_features = 0
    total_reconstructed_features = 0
    total_nodes = 0
    total_edges = 0

    for road_file in road_files:

        print(
            f"\n--- {road_file.name} ---"
        )

        roads = gpd.read_file(
            road_file
        )

        input_count = len(roads)

        total_input_features += (
            input_count
        )

        if roads.empty:

            print(
                "Empty road file — skipped."
            )

            continue

        # ----------------------------------------------------
        # Convert geographic coordinates to metres.
        # ----------------------------------------------------

        roads = roads.to_crs(
            METRIC_CRS
        )

        # ----------------------------------------------------
        # Split roads at intersections.
        # ----------------------------------------------------

        reconstructed = reconstruct_roads(
            roads
        )

        reconstructed_count = len(
            reconstructed
        )

        total_reconstructed_features += (
            reconstructed_count
        )

        print(
            f"Road features: "
            f"{input_count} → "
            f"{reconstructed_count}"
        )

        # ----------------------------------------------------
        # Build graph.
        # ----------------------------------------------------

        graph = build_graph(
            reconstructed
        )

        analyze_graph(
            graph
        )

        total_nodes += (
            graph.number_of_nodes()
        )

        total_edges += (
            graph.number_of_edges()
        )

    # ========================================================
    # Summary
    # ========================================================

    print("\n========================================")
    print("       RECONSTRUCTION SUMMARY")
    print("========================================")

    print(
        f"Input road features: "
        f"{total_input_features}"
    )

    print(
        f"Reconstructed segments: "
        f"{total_reconstructed_features}"
    )

    print(
        f"Total graph nodes: "
        f"{total_nodes}"
    )

    print(
        f"Total graph edges: "
        f"{total_edges}"
    )

    print(
        f"Metric CRS: "
        f"{METRIC_CRS}"
    )

    print(
        "\n===== NETWORK RECONSTRUCTION COMPLETE ====="
    )


if __name__ == "__main__":
    main()