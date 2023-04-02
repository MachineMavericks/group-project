# IMPORTS=
import copy

# DAY-BY-DAY EXTRACTION:
def day_n_graph(graph, dayn):
    # Filter the edges/nodes using their nodes' passages' and edges' travels' day:
    graph_cp = copy.deepcopy(graph)
    nodes_to_retain = []
    for node in graph.nodes:
        node_contains_day_n = False
        for passage in node.get_passages():
            if passage.get_day() == dayn:
                node_contains_day_n = True
                break
        if node_contains_day_n:
            nodes_to_retain.append(node)
    graph_cp.nodes = nodes_to_retain
    edges_to_retain = []
    for edge in graph.edges:
        edge_contains_day_n = False
        for travel in edge.get_travels():
            if travel.get_day() == dayn:
                edge_contains_day_n = True
                break
        if edge_contains_day_n:
            edges_to_retain.append(edge)
    graph_cp.edges = edges_to_retain
    return graph_cp
