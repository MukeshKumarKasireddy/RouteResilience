import networkx as nx


def edge_betweenness(graph):
    return nx.edge_betweenness_centrality(graph, weight="length")


def edge_connectivity_impact(graph):
    scores = {}
    base_components = nx.number_connected_components(graph)

    for u, v in graph.edges():
        test_graph = graph.copy()
        test_graph.remove_edge(u, v)
        new_components = nx.number_connected_components(test_graph)
        scores[(u, v)] = new_components - base_components

    return scores

def compute_criticality(graph):
    betweenness = edge_betweenness(graph)
    connectivity = edge_connectivity_impact(graph)

    scores = {}
    for edge in graph.edges():
        key = edge if edge in betweenness else (edge[1], edge[0])
        scores[edge] = {
            "betweenness": betweenness.get(key, betweenness.get((edge[1], edge[0]), 0)),
            "connectivity_impact": connectivity.get(edge, 0),
        }

    return scores


def rank_critical_edges(graph, top_n=5):
    scores = compute_criticality(graph)
    ranked = sorted(
        scores.items(),
        key=lambda item: (item[1]["connectivity_impact"], item[1]["betweenness"]),
        reverse=True,
    )
    return ranked[:top_n]
