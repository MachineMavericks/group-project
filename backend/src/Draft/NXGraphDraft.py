# IMPORTS=
import networkx.algorithms.community as nx_comm
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
import networkx as nx
from kneed import KneeLocator
import numpy as np

# MODELS=
from src.Models.NXGraph import NXGraph
from src.Models.Graph import Graph

# WARNINGS=
import warnings

warnings.filterwarnings("ignore")


# TODO: wrap duplicated code fragments to single helper function

# HEATMAP - NODES_WEIGHT=PASSAGE - OUTPUT=PLT:
def node_passages_plt_heatmap(nxgraph, save_png=False):
    # Position of the nodes:
    pos = {node: (nxgraph.nodes[node]['lon'], nxgraph.nodes[node]['lat']) for node in nxgraph.nodes}
    max_weight = max([nxgraph.nodes[node]['total_minutes'] for node in nxgraph.nodes])
    min_weight = min([nxgraph.nodes[node]['total_minutes'] for node in nxgraph.nodes])
    average_weight = sum([nxgraph.nodes[node]['total_minutes'] for node in nxgraph.nodes]) / len(nxgraph.nodes)
    median_weight = sorted([nxgraph.nodes[node]['total_minutes'] for node in nxgraph.nodes])[len(nxgraph.nodes) // 2]
    min_node_size = 1
    max_node_size = 4
    vmin = min_weight
    vmax = median_weight * 5
    # Add the legend:
    plt.colorbar(plt.cm.ScalarMappable(
        cmap=plt.cm.coolwarm,
        norm=plt.Normalize(vmin=vmin, vmax=vmax)),
        label="Node's train passages time (in minutes)"
    )
    # Plot the networkx nodes with a heatmap color:
    nx.draw_networkx_nodes(
        G=nxgraph,
        pos=pos,
        node_size=[
            min_node_size + (max_node_size - min_node_size) * (nxgraph.nodes[node]['total_minutes'] - min_weight) / (
                    max_weight - min_weight) for node in nxgraph.nodes],
        node_color=[nxgraph.nodes[node]['total_minutes'] for node in nxgraph.nodes],
        cmap=plt.cm.coolwarm,
        vmin=vmin,
        vmax=vmax)
    if save_png:
        # Save the figure as a png:
        plt.savefig("resources/data/output/node_passages_heatmap.png", dpi=1200)
    plt.show()


# HEATMAP - NODES_WEIGHT=TRAVELS - OUTPUT=PLT:
def edge_travels_plt_heatmap(nxgraph, save_png=False):
    # Position of the nodes:
    pos = {node: (nxgraph.nodes[node]['lon'], nxgraph.nodes[node]['lat']) for node in nxgraph.nodes}
    # Heatmap using the edges's weight as the color (use a red colormap)
    max_weight = max([nxgraph.edges[edge]['total_minutes'] for edge in nxgraph.edges])
    min_weight = min([nxgraph.edges[edge]['total_minutes'] for edge in nxgraph.edges])
    average_weight = sum([nxgraph.edges[edge]['total_minutes'] for edge in nxgraph.edges]) / len(nxgraph.edges)
    median_weight = sorted([nxgraph.edges[edge]['total_minutes'] for edge in nxgraph.edges])[len(nxgraph.edges) // 2]
    vmin = min_weight
    vmax = median_weight * 5
    # Add the legend:
    plt.colorbar(plt.cm.ScalarMappable(
        cmap=plt.cm.coolwarm,
        norm=plt.Normalize(vmin=vmin, vmax=vmax)),
        label="Edge's train travel time (in minutes)"
    )
    # Plot the networkx edges with a heatmap color:
    nx.draw_networkx_edges(
        G=nxgraph,
        pos=pos,
        edge_color=[nxgraph.edges[edge]['total_minutes'] for edge in nxgraph.edges],
        edge_cmap=plt.cm.coolwarm,
        edge_vmin=vmin,
        edge_vmax=vmax,
        width=0.2)
    if save_png:
        # Save the figure as a png:
        plt.savefig("resources/data/output/edge_travels_heatmap.png", dpi=1200)
    plt.show()


# DAY-BY-DAY (PLOT) - AT LEAST DAY N
def plt_atleast_day_n_nxgraph(nxgraph, day_n):
    # selected_nodes = [node for node in nxgraph.nodes if day_n in nxgraph.nodes[node]['working_days']]
    selected_edges = [edge for edge in nxgraph.edges if day_n in nxgraph.edges[edge]['working_days']]
    # Find nodes that are connected to selected edges:
    selected_nodes = []
    for edge in selected_edges:
        selected_nodes.append(edge[0])
        selected_nodes.append(edge[1])
    selected_nodes = list(set(selected_nodes))
    pos = {node: (nxgraph.nodes[node]['lon'], nxgraph.nodes[node]['lat']) for node in nxgraph.nodes}
    nx.draw_networkx_nodes(nxgraph, pos, nodelist=selected_nodes, node_size=2, node_color='r')
    nx.draw_networkx_edges(nxgraph, pos, edgelist=selected_edges, width=0.2, edge_color='black')
    plt.title("Rails working at least on day {}".format(day_n))
    plt.show()


# DAY-BY-DAY (PLOT) - ONLY DAY N
def plt_only_day_n_nxgraph(nxgraph, day_n):
    # selected_nodes = [node for node in nxgraph.nodes if day_n in nxgraph.nodes[node]['working_days'] and len(
    # nxgraph.nodes[node]['working_days']) == 1]
    selected_edges = [edge for edge in nxgraph.edges if
                      day_n in nxgraph.edges[edge]['working_days'] and len(nxgraph.edges[edge]['working_days']) == 1]
    # Find nodes that are connected to selected edges:
    selected_nodes = []
    for edge in selected_edges:
        selected_nodes.append(edge[0])
        selected_nodes.append(edge[1])
    selected_nodes = list(set(selected_nodes))
    pos = {node: (nxgraph.nodes[node]['lon'], nxgraph.nodes[node]['lat']) for node in nxgraph.nodes}
    nx.draw_networkx_nodes(nxgraph, pos, nodelist=selected_nodes, node_size=2, node_color='r')
    nx.draw_networkx_edges(nxgraph, pos, edgelist=selected_edges, width=0.2, edge_color='black')
    plt.title("Rails working only on day {}".format(day_n))
    plt.show()


# DAY-BY-DAY (PLOT) - AT LEAST DAYS N
def plt_atleast_days_n_nxgraph(nxgraph, days_n):
    # selected_nodes = [node for node in nxgraph.nodes if all(day_n in nxgraph.nodes[node]['working_days'] for day_n
    # in days_n)]
    selected_edges = [edge for edge in nxgraph.edges if
                      all(day_n in nxgraph.edges[edge]['working_days'] for day_n in days_n)]
    # Find nodes that are connected to selected edges:
    selected_nodes = []
    for edge in selected_edges:
        selected_nodes.append(edge[0])
        selected_nodes.append(edge[1])
    selected_nodes = list(set(selected_nodes))
    pos = {node: (nxgraph.nodes[node]['lon'], nxgraph.nodes[node]['lat']) for node in nxgraph.nodes}
    nx.draw_networkx_nodes(nxgraph, pos, nodelist=selected_nodes, node_size=2, node_color='r')
    nx.draw_networkx_edges(nxgraph, pos, edgelist=selected_edges, width=0.2, edge_color='black')
    plt.title("Rails working at least on days {}".format(days_n))
    plt.show()


# DAY-BY-DAY (PLOT) - ONLY DAYS N
def plt_only_days_n_nxgraph(nxgraph, days_n):
    # selected_nodes = [node for node in nxgraph.nodes if all(day_n in nxgraph.nodes[node]['working_days'] for day_n
    # in days_n) and len(nxgraph.nodes[node]['working_days']) == len(days_n)]
    selected_edges = [edge for edge in nxgraph.edges if
                      all(day_n in nxgraph.edges[edge]['working_days'] for day_n in days_n) and len(
                          nxgraph.edges[edge]['working_days']) == len(days_n)]
    # Find nodes that are connected to selected edges:
    selected_nodes = []
    for edge in selected_edges:
        selected_nodes.append(edge[0])
        selected_nodes.append(edge[1])
    selected_nodes = list(set(selected_nodes))
    pos = {node: (nxgraph.nodes[node]['lon'], nxgraph.nodes[node]['lat']) for node in nxgraph.nodes}
    nx.draw_networkx_nodes(nxgraph, pos, nodelist=selected_nodes, node_size=2, node_color='r')
    nx.draw_networkx_edges(nxgraph, pos, edgelist=selected_edges, width=0.2, edge_color='black')
    plt.title("Rails working only on days {}".format(days_n))
    plt.show()


# DAY-BY-DAY (PLOT) - AT LEAST DAYS N - 4x4
def plt_16_figure_atleastdays_nxgraph(nxgraph):
    # Plot a 4x4 figure with all combinations of days
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    for i in range(4):
        for j in range(4):
            days_n = [1, 2, 3, 4][:i + j + 1]
            # selected_nodes = [node for node in nxgraph.nodes if all(day_n in nxgraph.nodes[node]['working_days']
            # for day_n in days_n)]
            selected_edges = [edge for edge in nxgraph.edges if
                              all(day_n in nxgraph.edges[edge]['working_days'] for day_n in days_n)]
            # Find nodes that are connected to selected edges:
            selected_nodes = []
            for edge in selected_edges:
                selected_nodes.append(edge[0])
                selected_nodes.append(edge[1])
            selected_nodes = list(set(selected_nodes))
            pos = {node: (nxgraph.nodes[node]['lon'], nxgraph.nodes[node]['lat']) for node in nxgraph.nodes}
            nx.draw_networkx_nodes(nxgraph, pos, nodelist=selected_nodes, node_size=2, node_color='r', ax=axs[i, j])
            nx.draw_networkx_edges(nxgraph, pos, edgelist=selected_edges, width=0.2, edge_color='black', ax=axs[i, j])
            axs[i, j].set_title("Rails working at least on days {}".format(days_n))
    plt.show()


# DAY-BY-DAY (PLOT) - ONLY DAYS N - 4x4
def plt_16_figure_onlydays_nxgraph(nxgraph):
    # Plot a 4x4 figure with all combinations of days
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    for i in range(4):
        for j in range(4):
            days_n = [1, 2, 3, 4][:i + j + 1]
            # selected_nodes = [node for node in nxgraph.nodes if all(day_n in nxgraph.nodes[node]['working_days']
            # for day_n in days_n) and len(nxgraph.nodes[node]['working_days']) == len(days_n)]
            selected_edges = [edge for edge in nxgraph.edges if
                              all(day_n in nxgraph.edges[edge]['working_days'] for day_n in days_n) and len(
                                  nxgraph.edges[edge]['working_days']) == len(days_n)]
            # Find nodes that are connected to selected edges:
            selected_nodes = []
            for edge in selected_edges:
                selected_nodes.append(edge[0])
                selected_nodes.append(edge[1])
            selected_nodes = list(set(selected_nodes))
            pos = {node: (nxgraph.nodes[node]['lon'], nxgraph.nodes[node]['lat']) for node in nxgraph.nodes}
            nx.draw_networkx_nodes(nxgraph, pos, nodelist=selected_nodes, node_size=2, node_color='r', ax=axs[i, j])
            nx.draw_networkx_edges(nxgraph, pos, edgelist=selected_edges, width=0.2, edge_color='black', ax=axs[i, j])
            axs[i, j].set_title("Rails working only on days {}".format(days_n))
    plt.show()


# ...
def nodes_whose_centrality_degree_is_greater_than(nx_graph, threshold):
    return [node for (node, deg) in nx_graph.degree() if deg > threshold]


# ...
def spatial_clustering(nx_graph):
    pos = {node: (nx_graph.nodes[node]['lon'], nx_graph.nodes[node]['lat']) for node in nx_graph.nodes}

    Ks = range(2, 50)
    wss_values = []
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=0)
        km.fit(list(pos.values()))
        wss_values.append(km.inertia_)

    # Plot WSS values
    kn = KneeLocator(Ks, wss_values, curve='convex', direction='decreasing')
    elbow_k = kn.knee

    plt.plot(Ks, wss_values, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-cluster sum of squares (WSS)')
    plt.title('Elbow Method for Optimal k')
    plt.vlines(elbow_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()

    print(f"Optimal number of clusters (elbow point): {elbow_k}")

    km = KMeans(n_clusters=elbow_k).fit(list(pos.values()))
    clusters = km.predict(list(pos.values()))
    nx.draw_networkx_nodes(nx_graph, pos, node_size=1, node_color=clusters, cmap=plt.cm.jet)
    nx.draw_networkx_edges(nx_graph, pos, edge_color='black', width=0.2)

    plt.show()


def spectral_clustering(nx_graph, edge_weight='total_travels'):
    pos = {node: (nx_graph.nodes[node]['lon'], nx_graph.nodes[node]['lat']) for node in nx_graph.nodes}
    # create the affinity matrix
    affinity_matrix = np.zeros((len(nx_graph), len(nx_graph)))
    for i, u in enumerate(nx_graph.nodes()):
        for j, v in enumerate(nx_graph.nodes()):
            if i != j:
                affinity_matrix[i, j] = 1 / nx_graph[u][v][0][edge_weight] if nx_graph.has_edge(u, v) else 0

    # perform spectral clustering
    spectral = SpectralClustering(n_clusters=15, affinity='precomputed')
    spectral.fit(affinity_matrix)
    clusters = spectral.labels_

    # create a colormap
    cmap = plt.cm.get_cmap('jet', 10)

    # map cluster labels to colors
    edge_colors = [cmap(i) for i in clusters]

    # draw the graph
    # nx.draw_networkx_nodes(nx_graph, pos, node_size=1)
    nx.draw_networkx_edges(nx_graph, pos, width=0.2, edge_color=edge_colors)

    # show the plot
    plt.show()


# ...
"""
# ___________________________________________________________
     SURELY SUCKS BUT MAY BE USEFUL AT SOME POINT
# ___________________________________________________________

def betweenness_clustering(nx_graph):
    # Calculate betweenness centrality scores
    bet = nx.betweenness_centrality(nx_graph)

    # Cluster nodes based on betweenness centrality
    X = np.array(list(bet.values())).reshape(-1, 1)
    kmeans = KMeans(n_clusters=10).fit(X)
    clusters = kmeans.predict(X)

    # Calculate average betweenness centrality score for each cluster
    cluster_scores = {}
    for node, cluster in zip(bet.keys(), clusters):
        if cluster not in cluster_scores:
            cluster_scores[cluster] = []
        cluster_scores[cluster].append(bet[node])
    cluster_scores = {cluster: np.mean(scores) for cluster, scores in cluster_scores.items()}

    # Get the geographic positions of the nodes from the node attributes
    pos = {node[0]: (node[1]['lon'], node[1]['lat']) for node in nx_graph.nodes(data=True)}

    # Add a colorbar for the centrality scores
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(bet.values()), vmax=max(bet.values())))
    sm._A = []
    plt.colorbar(sm)

    # Draw the graph with nodes colored by cluster and node size proportional to betweenness centrality
    node_colors = [plt.cm.Reds(cluster_scores[cluster] / max(cluster_scores.values())) for cluster in clusters]
    node_sizes = [bet[node] * 500 for node in nx_graph.nodes()]

    nx.draw_networkx_nodes(nx_graph, pos=pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(nx_graph, pos=pos, edge_color='white', width=0.2)

    # Show the plot
    plt.show()

    # for i in range(10):
    #     print(f"Cluster {i}: {[nx_graph.nodes()[j] for j in range(len(clusters)) if clusters[j] == i]}")
"""


def get_node_cluster_info(nx_graph, labels):
    clusters = {}
    for node, label in zip(nx_graph.nodes, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node)

    avg_degrees = {}
    avg_deg = 0

    degree = dict(nx_graph.degree())
    for label, nodes in clusters.items():
        for node in nodes:
            avg_deg += degree[node]

        avg_deg /= len(clusters[label])
        avg_degrees[label] = avg_deg

    print(avg_degrees)


# ...
def find_communities(nx_graph, weight='mileage'):
    pos = {node: (nx_graph.nodes[node]['lon'], nx_graph.nodes[node]['lat']) for node in nx_graph.nodes}
    communities = nx_comm.louvain_communities(nx_graph, weight=weight)
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(nx_graph, pos, nodelist=list(community),
                               node_color=plt.cm.tab20(i),
                               node_size=2,
                               )
    # nx.draw_networkx_edges(nx_graph, pos=pos, edge_color='white', width=0.2)
    plt.show()


# TESTING
def main():
    nxgraph = NXGraph(pickle_path="../../resources/data/output/chinese_railway/graph.pickle", dataset_number=1, day=2)
    spatial_clustering(nxgraph)


if __name__ == "__main__":
    main()
