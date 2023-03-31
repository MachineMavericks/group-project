import networkx as nx
import networkx.algorithms.community as nx_comm

from src.Models import NXGraph as NXG
from src.Models import Graph as G
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
from kneed import KneeLocator

import warnings

warnings.filterwarnings("ignore")

graph = G.Graph('../../resources/data/input/railway.csv')
nxg = NXG.nxgraph_loader(graph)


def nodes_whose_centrality_degree_is_greater_than(nx_graph, threshold):
    return [node for (node, deg) in nx_graph.degree() if deg > threshold]


def spatial_clustering(nx_graph):
    pos = {node[0]: (node[1]['lon'], node[1]['lat']) for node in nx_graph.nodes(data=True)}

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


def find_communities(nx_graph):
    pos = {node[0]: (node[1]['lon'], node[1]['lat']) for node in nx_graph.nodes(data=True)}

    communities = nx_comm.louvain_communities(nx_graph)

    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(nx_graph, pos, nodelist=list(community),
                               node_color=plt.cm.tab20(i),
                               node_size=2,
                               )
    # nx.draw_networkx_edges(nx_graph, pos=pos, edge_color='white', width=0.2)
    plt.show()


# TESTING
def main():
    find_communities(nxg)


if __name__ == "__main__":
    main()
