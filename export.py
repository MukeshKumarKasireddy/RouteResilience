import json

import networkx as nx


def edge_to_dict(edge, graph):
    u, v = edge
    data = graph.get_edge_data(u, v, default={})
    return {
        "road_id": data.get("road_id"),
        "road_type": data.get("road_type"),
        "from_node": u,
        "to_node": v,
        "length": data.get("length"),
    }


def build_result(graph, ranked_critical_edges, scenario_name, affected_edges, before_after_metrics):
    critical_segments = [
        {
            "edge": edge_to_dict(edge, graph),
            "betweenness": scores["betweenness"],
            "connectivity_impact": scores["connectivity_impact"],
        }
        for edge, scores in ranked_critical_edges
    ]

    affected = [edge_to_dict(edge, graph) for edge in affected_edges]

    return {
        "graph": nx.node_link_data(graph, edges="edges"),
        "critical_road_segments": critical_segments,
        "failure_scenario": scenario_name,
        "affected_roads": affected,
        "metrics": before_after_metrics,
    }


def save_result(result, path):
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
