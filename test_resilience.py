import os


import build_graph
import criticality
import simulation
import metrics

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_road_network.geojson")


def load_sample_graph():
    return build_graph.build_graph_from_file(SAMPLE_PATH)


def test_rank_critical_edges():
    graph = load_sample_graph()
    ranked = criticality.rank_critical_edges(graph, top_n=3)
    assert len(ranked) == 3
    for edge, scores in ranked:
        assert "betweenness" in scores
        assert "connectivity_impact" in scores


def test_single_road_failure_removes_edge():
    graph = load_sample_graph()
    edge = list(graph.edges())[0]
    after_graph, affected = simulation.single_road_failure(graph, edge)
    assert after_graph.number_of_edges() == graph.number_of_edges() - 1
    assert affected == [edge]


def test_critical_road_failure():
    graph = load_sample_graph()
    ranked = criticality.rank_critical_edges(graph, top_n=3)
    after_graph, affected = simulation.critical_road_failure(graph, ranked)
    assert len(affected) == 1
    assert after_graph.number_of_edges() == graph.number_of_edges() - 1


def test_compare_before_after():
    graph = load_sample_graph()
    edge = list(graph.edges())[0]
    after_graph, _ = simulation.single_road_failure(graph, edge)
    result = metrics.compare_before_after(graph, after_graph)
    assert result["before"]["edges"] == graph.number_of_edges()
    assert result["after"]["edges"] == after_graph.number_of_edges()
