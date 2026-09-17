import networkx as nx


def total_edge_length(graph):
    return sum(data.get("length", 0) for _, _, data in graph.edges(data=True))


def largest_component_size(graph):
    if graph.number_of_nodes() == 0:
        return 0
    return len(max(nx.connected_components(graph), key=len))


def average_shortest_path(graph):
    if graph.number_of_nodes() < 2:
        return None

    largest = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(largest)

    if subgraph.number_of_nodes() < 2:
        return None

    return nx.average_shortest_path_length(subgraph, weight="length")


def compute_metrics(graph):
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "connected_components": nx.number_connected_components(graph),
        "largest_component_size": largest_component_size(graph),
        "total_road_length": total_edge_length(graph),
        "average_shortest_path": average_shortest_path(graph),
    }


def compare_before_after(before_graph, after_graph):
    before = compute_metrics(before_graph)
    after = compute_metrics(after_graph)

    return {
        "before": before,
        "after": after,
        "nodes_disconnected": before["largest_component_size"] - after["largest_component_size"],
        "road_length_lost": before["total_road_length"] - after["total_road_length"],
        "connected_components_increase": after["connected_components"] - before["connected_components"],
    }
