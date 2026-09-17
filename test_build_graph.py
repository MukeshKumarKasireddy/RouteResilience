import os


import build_graph

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_road_network.geojson")


def test_build_graph_from_sample():
    graph = build_graph.build_graph_from_file(SAMPLE_PATH)
    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 7


def test_edges_have_length():
    graph = build_graph.build_graph_from_file(SAMPLE_PATH)
    for _, _, data in graph.edges(data=True):
        assert data["length"] > 0


def test_save_and_load_graph(tmp_path):
    graph = build_graph.build_graph_from_file(SAMPLE_PATH)
    out_path = tmp_path / "graph.json"
    build_graph.save_graph_json(graph, out_path)
    loaded = build_graph.load_graph_json(out_path)
    assert loaded.number_of_nodes() == graph.number_of_nodes()
    assert loaded.number_of_edges() == graph.number_of_edges()
