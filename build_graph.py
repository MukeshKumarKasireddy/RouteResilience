import json
import math

import networkx as nx


def load_road_network(path):
    with open(path, "r") as f:
        return json.load(f)


def _node_id(lon, lat, precision=6):
    return f"{round(lon, precision)},{round(lat, precision)}"


def haversine_length(coords):
    r = 6371000
    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
        total += 2 * r * math.asin(min(1, math.sqrt(a)))
    return total


def build_graph(feature_collection):
    graph = nx.Graph()

    for feature in feature_collection["features"]:
        geometry = feature["geometry"]
        if geometry["type"] != "LineString":
            continue

        coords = geometry["coordinates"]
        props = feature.get("properties", {})

        start = coords[0]
        end = coords[-1]
        start_id = _node_id(*start)
        end_id = _node_id(*end)

        graph.add_node(start_id, lon=start[0], lat=start[1])
        graph.add_node(end_id, lon=end[0], lat=end[1])

        length = props.get("length")
        if length is None:
            length = haversine_length(coords)

        graph.add_edge(
            start_id,
            end_id,
            road_id=props.get("road_id"),
            road_type=props.get("road_type", "unknown"),
            length=length,
            confidence=props.get("confidence"),
            geometry=coords,
        )

    return graph


def build_graph_from_file(path):
    feature_collection = load_road_network(path)
    return build_graph(feature_collection)


def save_graph_json(graph, path):
    data = nx.node_link_data(graph, edges="edges")
    with open(path, "w") as f:
        json.dump(data, f)


def load_graph_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="edges")
