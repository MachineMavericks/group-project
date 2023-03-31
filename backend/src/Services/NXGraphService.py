# IMPORTS=
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as go

# MODELS=
from src.Models.NXGraph import NXGraph

# MATPLOTLIB:
# HEATMAP - NODES_WEIGHT=PASSAGE - OUTPUT=PLT:
def node_passages_plt_heatmap(nxgraph, save_png=False):
    # Position of the nodes:
    pos = {node: (nxgraph.nodes[node]['lon'], nxgraph.nodes[node]['lat']) for node in nxgraph.nodes}
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
    plt.show()
# HEATMAP - NODES_WEIGHT=TRAVELS - OUTPUT=PLT:
def edge_travels_plt_heatmap(nxgraph, save_png=False):
    # Position of the nodes:
    pos = {node: (nxgraph.nodes[node]['lon'], nxgraph.nodes[node]['lat']) for node in nxgraph.nodes}
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
    plt.show()

# PLOTLY:
# TODO: HEATMAP - NODES_WEIGHT=PASSAGE - OUTPUT=PLOTLY_HTML:
def plotly_heatmap_nodes(nxgraph, save_html=False):
    pass