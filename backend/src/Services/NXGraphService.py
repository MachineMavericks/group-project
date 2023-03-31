import matplotlib.pyplot as plt
import networkx as nx

from src.Models import NXGraph as NXG
from src.Models import Graph as G

graph = G.Graph('resources/data/input/railway.csv')
nxgraph = NXG.nxgraph_loader(graph)

def nodes_whose_centrality_degree_is_greater_than(graph, degree):
    # TODO.
    # return nodes[]
    pass

# Heatmap using the nodes's weight as the color (use a red colormap)
def node_passages_plt_heatmap(nxgraph, pos, save_png=False):
    max_weight = max([nxgraph.nodes[node]['weight'] for node in nxgraph.nodes])
    min_weight = min([nxgraph.nodes[node]['weight'] for node in nxgraph.nodes])
    average_weight = sum([nxgraph.nodes[node]['weight'] for node in nxgraph.nodes]) / len(nxgraph.nodes)
    median_weight = sorted([nxgraph.nodes[node]['weight'] for node in nxgraph.nodes])[len(nxgraph.nodes) // 2]
    min_node_size = 1
    max_node_size = 4
    vmin = min_weight
    vmax = median_weight*5
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
        node_size=[min_node_size + (max_node_size - min_node_size) * (nxgraph.nodes[node]['weight'] - min_weight) / (max_weight - min_weight) for node in nxgraph.nodes],
        node_color=[nxgraph.nodes[node]['weight'] for node in nxgraph.nodes],
        cmap=plt.cm.coolwarm,
        vmin=vmin,
        vmax=vmax)
    if save_png:
        # Save the figure as a png:
        plt.savefig("resources/data/output/node_passages_heatmap.png", dpi=1200)

def edge_travels_plt_heatmap(nxgraph, pos, save_png=False):
    # Heatmap using the edges's weight as the color (use a red colormap)
    max_weight = max([nxgraph.edges[edge]['weight'] for edge in nxgraph.edges])
    min_weight = min([nxgraph.edges[edge]['weight'] for edge in nxgraph.edges])
    average_weight = sum([nxgraph.edges[edge]['weight'] for edge in nxgraph.edges]) / len(nxgraph.edges)
    median_weight = sorted([nxgraph.edges[edge]['weight'] for edge in nxgraph.edges])[len(nxgraph.edges) // 2]
    vmin = min_weight
    vmax = median_weight*5
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
        edge_color=[nxgraph.edges[edge]['weight'] for edge in nxgraph.edges],
        edge_cmap=plt.cm.coolwarm,
        edge_vmin=vmin,
        edge_vmax=vmax,
        width=0.2)
    if save_png:
        # Save the figure as a png:
        plt.savefig("resources/data/output/edge_travels_heatmap.png", dpi=1200)