import networkx as nx
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# MODELS=
from src.Models.NXGraph import NXGraph
from src.Models.Graph import Graph

# WARNINGS=
import warnings

warnings.filterwarnings("ignore")


# UTILS=
def plot_graph(nx_graph, title='Default graph'):
    pos = {node: (nx_graph.nodes[node]['lon'], nx_graph.nodes[node]['lat']) for node in nx_graph.nodes}
    nx.draw_networkx_nodes(nx_graph, pos=pos, node_size=1)
    nx.draw_networkx_edges(nx_graph, pos=pos, edge_color='black', width=0.2)

    plt.title(title)
    plt.show()


def largest_connected_component_ratio(original_graph, attacked_graph):
    og_cc, cc = nx.connected_components(original_graph), nx.connected_components(attacked_graph)
    og_lcc, lcc = max(og_cc, key=len), max(cc, key=len)

    return len(lcc) / len(og_lcc)


def average_shortest_path_length_ratio(original_graph, attacked_graph):
    original_avg_spl = nx.average_shortest_path_length(original_graph)
    # shortest path computation is infinite if graph is disconnected
    try:
        new_avg_spl = nx.average_shortest_path_length(attacked_graph)

    except nx.NetworkXError:
        new_avg_spl = nx.diameter(original_graph)

    return new_avg_spl / original_avg_spl


# ATTACKS=
def random_attack(nx_graph, fraction=0.1, iterations=100):
    lcc_ratio = 0
    avg_sp_ratio = 0

    for i in tqdm(range(iterations)):
        nx_graph_copy = nx_graph.copy()
        nodes = list(nx_graph_copy.nodes())
        steps = 0
        num_nodes_to_remove = int(len(nodes) * fraction)

        while nodes and steps < num_nodes_to_remove:
            node = random.choice(nodes)
            nodes.remove(node)
            nx_graph_copy.remove_node(node)
            steps += 1

        # evaluate
        lcc_ratio += largest_connected_component_ratio(nx_graph, nx_graph_copy)
        # avg_sp_ratio += average_shortest_path_length_ratio(nx_graph, nx_graph_copy)

    print("Average largest connected component ratio: ", lcc_ratio / iterations)
    # print("Average average shortest path length ratio: ", avg_sp_ratio / iterations)


# TESTING
def main():
    # graph = Graph("../../resources/data/input/railway.csv", save_csvs=True, output_dir="../../resources/data/output/")
    # nxg = NXGraph(graph)
    # nx.write_gml(nxg, "../../resources/data/output/nx_graph.gml", stringizer=str)
    nxg = nx.read_gml("../../resources/data/output/nx_graph.gml")
    random_attack(nxg)


if __name__ == "__main__":
    main()
