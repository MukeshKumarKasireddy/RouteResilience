def single_road_failure(graph, edge):
    after_graph = graph.copy()
    if after_graph.has_edge(*edge):
        after_graph.remove_edge(*edge)
    return after_graph, [edge]


def geographic_zone_failure(graph, bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    after_graph = graph.copy()
    removed = []

    for u, v in list(graph.edges()):
        u_lon = graph.nodes[u]["lon"]
        u_lat = graph.nodes[u]["lat"]
        v_lon = graph.nodes[v]["lon"]
        v_lat = graph.nodes[v]["lat"]

        u_inside = min_lon <= u_lon <= max_lon and min_lat <= u_lat <= max_lat
        v_inside = min_lon <= v_lon <= max_lon and min_lat <= v_lat <= max_lat

        if u_inside or v_inside:
            after_graph.remove_edge(u, v)
            removed.append((u, v))

    return after_graph, removed


def critical_road_failure(graph, ranked_critical_edges):
    if not ranked_critical_edges:
        return graph.copy(), []

    edge = ranked_critical_edges[0][0]
    return single_road_failure(graph, edge)
