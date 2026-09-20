import os
import sys

import build_graph
import criticality
import simulation
import metrics
import export


def run(input_path, output_path, scenario="critical"):
    graph = build_graph.build_graph_from_file(input_path)

    ranked = criticality.rank_critical_edges(graph, top_n=5)

    if scenario == "single":
        edge = list(graph.edges())[0]
        after_graph, affected_edges = simulation.single_road_failure(graph, edge)
    elif scenario == "zone":
        bbox = (78.40, 17.40, 78.41, 17.405)
        after_graph, affected_edges = simulation.geographic_zone_failure(graph, bbox)
    else:
        after_graph, affected_edges = simulation.critical_road_failure(graph, ranked)

    before_after = metrics.compare_before_after(graph, after_graph)

    result = export.build_result(graph, ranked, scenario, affected_edges, before_after)
    export.save_result(result, output_path)

    return result


if __name__ == "__main__":
    default_input = os.path.join(os.path.dirname(__file__), "sample_road_network.geojson")
    default_output = os.path.join(os.path.dirname(__file__), "results.json")

    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output
    scenario = sys.argv[3] if len(sys.argv) > 3 else "critical"

    run(input_path, output_path, scenario)
    print(f"saved results to {output_path}")
