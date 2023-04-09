# IMPORTS=
import networkx as nx
from itertools import permutations
import networkx.algorithms.community as nx_comm

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from scipy.special import factorial

# MODELS=
from src.Models.NXGraph import NXGraph
from src.Models.Graph import Graph

# WARNINGS=
import warnings

warnings.filterwarnings("ignore")


# FIGURE LAYOUT UPDATE FUNCTION:
def fig_update_layout(fig):
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"))
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'modebar_bgcolor': 'rgba(0, 0, 0, 0)',
        'margin': dict(l=0, r=0, t=30, b=0),
        'coloraxis_colorbar': dict(titlefont=dict(color="white"), tickfont=dict(color="white"))
    })
    fig.update_layout(title_font_color="white",
                      title_font_size=20,
                      title_x=0.5)


# DEFAULT PLOTTING OF NODES:
def plotly_default(pickle_path, day=None, output_path=None):
    nxgraph = NXGraph(pickle_path=pickle_path, dataset_number=1,
                      day=int(day) if day is not None and day != "" else None)
    # NODES DATAFRAME:
    nodes = []
    for node in nxgraph.nodes(data=True):
        id = int(node[0])
        lat = node[1]['lat']
        lon = node[1]['lon']
        total_passages = node[1]['total_passages']
        total_minutes = node[1]['total_minutes']
        nodes.append([id, lat, lon, total_passages, total_minutes])
    df_nodes = pd.DataFrame(nodes, columns=['Node ID', 'Latitude', 'Longitude', 'Total passages', 'Total minutes'])
    # EDGES DATAFRAME:
    edges = []
    for edge in nxgraph.edges(data=True):
        from_id = edge[0]
        dest_id = edge[1]
        from_lat = np.round(edge[2]['fromLat'], 2)
        from_lon = np.round(edge[2]['fromLon'], 2)
        dest_lat = np.round(edge[2]['destLat'], 2)
        dest_lon = np.round(edge[2]['destLon'], 2)
        total_travels = edge[2]['total_travels']
        total_minutes = edge[2]['total_minutes']
        edges.append([from_id, dest_id, from_lat, from_lon, dest_lat, dest_lon, total_travels, total_minutes])
    df_edges = pd.DataFrame(edges, columns=['Source Node', 'Destination Node', 'Source Latitude', 'Source Longitude',
                                            'Destination Latitude', 'Destination Longitude', 'Total travels',
                                            'Total minutes'])
    # CREATE NEW FIGURE:
    fig = px.scatter_mapbox(df_nodes[df_nodes["Node ID"] == -1], lat="Latitude", lon="Longitude", hover_name="Node ID",
                            hover_data=["Total passages", "Total minutes"], zoom=3.5, mapbox_style="open-street-map",
                            height=800, center=dict(lat=36, lon=117))
    # EDGES: Add edges as disconnected lines in a single trace
    edges_x = []
    edges_y = []
    for edge in nxgraph.edges(data=True):
        x0 = edge[2]['fromLon']
        y0 = edge[2]['fromLat']
        x1 = edge[2]['destLon']
        y1 = edge[2]['destLat']
        edges_x.append(x0)
        edges_x.append(x1)
        edges_x.append(None)
        edges_y.append(y0)
        edges_y.append(y1)
        edges_y.append(None)
    fig.add_trace(go.Scattermapbox(
        lat=edges_y,
        lon=edges_x,
        mode='lines',
        line=dict(width=1, color='grey'),
        hoverinfo='text',
        hovertext="Edge from " + df_edges['Source Node'].astype(str) + " to " + df_edges['Destination Node'].astype(
            str) + "<br>Travels: " + df_edges['Total travels'].astype(str) + "<br>Minutes: " + df_edges[
                      'Total minutes'].astype(str),
        name="Edges"
    ))
    # NODES:
    fig.add_scattermapbox(
        lat=df_nodes['Latitude'],
        lon=df_nodes['Longitude'],
        mode='markers',
        marker=dict(size=5, color='red'),
        hoverinfo='text',
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
        hovertext="Node n°" + df_nodes['Node ID'].astype(str) + "<br>Position: (Lat=" + df_nodes['Latitude'].astype(
            str) + ", Lon=" + df_nodes['Longitude'].astype(str) + ")<br>Total passages: " + df_nodes[
                      'Total passages'].astype(str) + "<br>Total minutes: " + df_nodes['Total minutes'].astype(str),
    )
    # GLOBAL SETTINGS:
    fig_update_layout(fig)
    # TITLE:
    fig.update_layout(title_text=f"Default plotly window " + (
        "(day=" + str(day) if day is not None and day != "" else "") + ", nodes=" + str(
        len(nxgraph.nodes())) + ", edges=" + str(len(nxgraph.edges())) + ")")
    # OUTPUT:
    if output_path is not None:
        fig.write_html(output_path)
    return fig


# PLOTLY HEATMAP:
def plotly_heatmap(pickle_path, day=None, component=None, metric="total_minutes", output_path=None):
    nxgraph = NXGraph(pickle_path=pickle_path, dataset_number=1,
                      day=int(day) if day is not None and day != "" else None)
    metric_name = ""
    var_factor = 3
    # DEFAULT:
    component = "node" if component is None else component
    metric = "total_minutes" if metric is None else metric
    # SPECIFIC:
    if component == "node":
        if metric == "total_minutes":
            metric_name = "Total minutes"
        elif metric == "total_passages":
            metric_name = "Total passages"
        elif metric == "degree_centrality":
            metric_name = "Degree centrality"
            print("Gathering degree centralities...")
            degree_centrality = nx.degree_centrality(nxgraph)
            nx.set_node_attributes(nxgraph, degree_centrality, 'degree_centrality')
        elif metric == "betweenness_centrality":
            metric_name = "Betweenness centrality"
            print("Gathering betweenness centralities...")
            betweenness_centrality = nx.betweenness_centrality(nxgraph)
            nx.set_node_attributes(nxgraph, betweenness_centrality, 'betweenness_centrality')
        elif metric == "closeness_centrality":
            metric_name = "Closeness centrality"
            print("Gathering closeness centralities...")
            closeness_centrality = nx.closeness_centrality(nxgraph)
            nx.set_node_attributes(nxgraph, closeness_centrality, 'closeness_centrality')
        else:
            raise NotImplementedError("Metric not implemented")
        min_metric = min([node[1][metric] for node in nxgraph.nodes(data=True)])
        max_metric = max([node[1][metric] for node in nxgraph.nodes(data=True)])
        median_metric = np.median([node[1][metric] for node in nxgraph.nodes(data=True)])
        avg_metric = np.mean([node[1][metric] for node in nxgraph.nodes(data=True)])
        vmin = min_metric
        vmax = median_metric + var_factor * (median_metric - min_metric)
        data = []
        for node in nxgraph.nodes(data=True):
            id = node[0]
            lat = np.round(node[1]['lat'], 2)
            lon = np.round(node[1]['lon'], 2)
            met = node[1][metric]
            data.append([id, lat, lon, met])
        df = pd.DataFrame(data, columns=['Node ID', 'Latitude', 'Longitude', metric_name])
        fig = px.density_mapbox(df, lat='Latitude', lon='Longitude', z=metric_name, radius=10,
                                center=dict(lat=36, lon=117), zoom=3.5, mapbox_style="open-street-map", height=800,
                                range_color=[vmin, vmax], hover_name='Node ID',
                                hover_data=['Latitude', 'Longitude', metric_name])
    elif component == "edge":
        raise NotImplementedError("Edge heatmap not implemented yet.")
    # GLOBAL SETTINGS:
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"))
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'modebar_bgcolor': 'rgba(0, 0, 0, 0)',
        'margin': dict(l=0, r=0, t=30, b=0),
        'coloraxis_colorbar': dict(titlefont=dict(color="white"), tickfont=dict(color="white"))
    })
    fig.update_layout(title_text=f"Heatmap: Component={component}, Metric={metric_name}, Day={day}",
                      title_font_color="white",
                      title_font_size=20,
                      title_x=0.5)
    # WRITE HTML FILE:
    if output_path is not None:
        fig.write_html(output_path)
    return fig


# PLOTLY RESILIENCE:
# helper functions:
def largest_connected_component_ratio(original_graph, attacked_graph):
    og_cc, cc = nx.connected_components(original_graph), nx.connected_components(attacked_graph)
    og_lcc, lcc = max(og_cc, key=len), max(cc, key=len)

    return len(lcc) / len(og_lcc)


def global_efficiency_weighted(graph, weight='mileage'):
    n = len(graph)
    denom = n * (n - 1)
    if denom != 0:
        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=weight))
        g_eff = sum(1. / shortest_paths[u][v] if v in shortest_paths[u] and shortest_paths[u][v] != 0 else 0
                    for u, v in permutations(graph, 2)) / denom
    else:
        g_eff = 0
    return g_eff


def global_efficiency_ratio(original_graph, attacked_graph):
    return global_efficiency_weighted(attacked_graph) / global_efficiency_weighted(original_graph)


# main function: TODO: add edges (red/white) + add sub functions for duplicated code
def plotly_resilience(pickle_path, day=None, strategy="targetted", component="node", metric="degree_centrality",
                      fraction="0.01", output_path=None):
    nxgraph = NXGraph(pickle_path=pickle_path, dataset_number=1,
                      day=int(day) if day is not None and day != "" else None)
    # DEFAULT:
    strategy = "targetted" if strategy is None else strategy
    component = "node" if component is None else component
    metric = "degree_centrality" if metric is None else metric
    fraction = "0.01" if fraction is None else fraction
    fraction = float(fraction)
    if strategy == "targetted":
        if component == "node":
            nx_graph_copy = nxgraph.copy()
            nodes_to_remove = int(len(nx_graph_copy) * fraction)

            metric_dict = {}
            if metric == 'degree_centrality':
                metric_dict = nx.degree_centrality(nx_graph_copy)
                nx.set_node_attributes(nx_graph_copy, metric_dict, 'degree_centrality')
                nx.set_node_attributes(nxgraph, metric_dict, 'degree_centrality')
            elif metric == 'betweenness_centrality':
                metric_dict = nx.betweenness_centrality(nx_graph_copy)
                nx.set_node_attributes(nx_graph_copy, metric_dict, 'betweenness_centrality')
                nx.set_node_attributes(nxgraph, metric_dict, 'betweenness_centrality')

            sorted_metric_dict = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:nodes_to_remove]
            for node, degree in sorted_metric_dict:
                nx_graph_copy.remove_node(node)

            # evaluate
            lcc = largest_connected_component_ratio(nxgraph, nx_graph_copy)
            global_eff = global_efficiency_ratio(nxgraph, nx_graph_copy)
            print("Largest connected component ratio: ", lcc)
            print("Global efficiency ratio: ", global_eff)
        elif component == "edge":
            raise NotImplementedError("Targetted edge resilience not implemented yet.")
    elif strategy == "random":
        if component == "node":
            raise NotImplementedError("Random node resilience not implemented yet.")
        elif component == "edge":
            raise NotImplementedError("Random edge resilience not implemented yet.")
    # PLOT:
    init_nodes = []
    for node in nxgraph.nodes(data=True):
        id = node[0]
        lat = np.round(node[1]['lat'], 2)
        lon = np.round(node[1]['lon'], 2)
        value = node[1][metric]
        init_nodes.append([id, lat, lon, value])
    destroyed_nodes = []
    for node in nxgraph.nodes(data=True):
        if node[0] in nx_graph_copy.nodes():
            continue
        id = node[0]
        lat = np.round(node[1]['lat'], 2)
        lon = np.round(node[1]['lon'], 2)
        value = node[1][metric]
        destroyed_nodes.append([id, lat, lon, value])
    init_df = pd.DataFrame(init_nodes, columns=['Node ID', 'Latitude', 'Longitude', metric])
    destroyed_df = pd.DataFrame(destroyed_nodes, columns=['Node ID', 'Latitude', 'Longitude', metric])
    fig = px.scatter_mapbox(init_df[init_df["Node ID"] == -1], lat='Latitude', lon='Longitude',
                            center=dict(lat=36, lon=117), zoom=3.5, mapbox_style="open-street-map", height=800)
    fig.add_scattermapbox(lat=init_df['Latitude'], lon=init_df['Longitude'], mode='markers',
                          marker=dict(size=3, color='blue'),
                          name="Nodes", hoverinfo="text",
                          hovertext="Node n°" + init_df['Node ID'].astype(str) + "<br>" + metric + ": " + init_df[
                              metric].astype(str))
    fig.add_scattermapbox(lat=destroyed_df['Latitude'], lon=destroyed_df['Longitude'], mode='markers',
                          marker=dict(size=6, color='red'),
                          name="Destroyed nodes", hoverinfo="text",
                          hovertext="Node n°" + destroyed_df['Node ID'].astype(str) + "<br>" + metric + ": " +
                                    destroyed_df[metric].astype(str))
    # GLOBAL SETTINGS:
    fig_update_layout(fig)
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"))
    fig.update_layout(
        title_text=f"Resilience: Strategy={strategy}, Component={component}, Metric={metric}, Fraction={fraction}, Day={day}",
        title_font_color="white",
        title_font_size=20,
        title_x=0.5)
    # WRITE HTML FILE:
    if output_path is not None:
        fig.write_html(output_path)


def plotly_clustering(pickle_path, day=None, algorithm='Euclidian k-mean', output_path=None):
    nxgraph = NXGraph(pickle_path=pickle_path, dataset_number=1,
                      day=int(day) if day is not None and day != "" else None)
    # DEFAULT:
    algorithm = "Euclidian k-mean" if algorithm is None else algorithm
    communities = {}
    if algorithm == "Euclidian k-mean":
        pos = {node: (nxgraph.nodes[node]['lon'], nxgraph.nodes[node]['lat']) for node in nxgraph.nodes}
        Ks = range(2, 50)
        wss_values = []
        for k in Ks:
            km = KMeans(n_clusters=k, random_state=0)
            km.fit(list(pos.values()))
            wss_values.append(km.inertia_)

        kn = KneeLocator(Ks, wss_values, curve='convex', direction='decreasing')
        elbow_k = kn.knee
        km = KMeans(n_clusters=elbow_k).fit(list(pos.values()))
        km_labels = km.predict(list(pos.values()))

        # Create a list of arrays, where each array contains the nodes in a cluster
        communities = [[] for _ in range(elbow_k)]
        for i, node in enumerate(list(pos.keys())):
            communities[km_labels[i]].append(node)

    elif algorithm == "Louvain":
        communities = nx_comm.louvain_communities(nxgraph, weight='mileage')

    # Create a list to store the data
    data = []
    for i, comm in enumerate(communities):
        for node in comm:
            lat = np.round(nxgraph.nodes[node]['lat'], 2)
            lon = np.round(nxgraph.nodes[node]['lon'], 2)
            data.append([node, lat, lon, i])

    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=['Node ID', 'Latitude', 'Longitude', 'Community'])
    # Create a list of colors for each community
    colors = px.colors.qualitative.Safe * (len(df['Community'].unique()) // len(px.colors.qualitative.Safe) + 1)

    # Map community IDs to colors
    df['Color'] = df['Community'].apply(lambda x: colors[x % len(colors)])

    # Create the plot using scatter_mapbox
    fig = px.scatter_mapbox(df[df["Node ID"] == -1], lat='Latitude', lon='Longitude',
                            center=dict(lat=36, lon=117), zoom=3.5,
                            mapbox_style="open-street-map", height=800)

    fig.add_scattermapbox(lat=df['Latitude'], lon=df['Longitude'], mode='markers',
                          marker=dict(size=5, color=df['Color']),
                          name="Nodes", hoverinfo="text",
                          hovertext="Node n°" + df['Node ID'].astype(str) + "<br>" + 'Cluster n°'
                                    + df['Community'].astype(str))

    # GLOBAL SETTINGS:
    fig_update_layout(fig)
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"))
    fig.update_layout(
        title_text=f"Clustering: Algorithm={algorithm}",
        title_font_color="white",
        title_font_size=20,
        title_x=0.5)
    # WRITE HTML FILE:
    if output_path is not None:
        fig.write_html(output_path)


def plotly_small_world(pickle_path, day=None, output_path=None):
    nxgraph = NXGraph(pickle_path=pickle_path, dataset_number=1,
                      day=int(day) if day is not None and day != "" else None)
    fig = make_subplots(rows=2, cols=2)
    fig.update_layout(title_text=f"Small-World Features " + (
        "(day " + str(day) + ")" if day is not None and day != "" else ""))
    nbins = 100
    nb_nd = nxgraph.number_of_nodes()
    # SHORTEST PATH HISTOGRAM -> small diameter
    shortest_paths = list(dict(list(dict(nx.all_pairs_shortest_path_length(nxgraph)).values())[0]).values())
    fig.add_trace(go.Histogram(x=shortest_paths, nbinsx=nbins, name="Shortest Path Length Histogram"), row=1, col=1)
    path_length_mean = sum(shortest_paths)/len(shortest_paths)
    annot_lenght = "Mean = " + str(round(path_length_mean, 2))
    fig.add_vline(x=path_length_mean, line_dash="dot", annotation_text=annot_lenght, annotation_position="top right",
                  row=1, col=1)
    # DEGREE HISTOGRAM -> some high degree nodes
    degrees = list(dict(nx.degree(nxgraph)).values())
    fig.add_trace(go.Histogram(x=degrees, nbinsx=nbins, name="Degree Histogram"), row=1, col=2)
    # CLUSTERING COEFFICIENT HISTOGRAM -> highly clustered
    digraph = nx.DiGraph(nxgraph) # nx.clustering not implemented for multigraphs
    clustering_coeffs = list(dict(nx.clustering(digraph)).values())
    fig.add_trace(go.Histogram(x=clustering_coeffs, nbinsx=nbins, name="Clustering Coefficients Histogram"),
                  row=2, col=1)
    clust_coeff_mean = sum(clustering_coeffs)/len(clustering_coeffs)
    annot_clust = "Network CC = " + str(round(clust_coeff_mean, 2))
    fig.add_vline(x=clust_coeff_mean, line_dash="dot", annotation_text=annot_clust, annotation_position="top right",
                  row=2, col=1)
    # DEGREE DISTRIBUTION HISTOGRAM AND POISON DISTRIBUTION -> follows Poisson
    degrees_nb = {}
    for i in range(100):
        degrees_nb.update({i: 0})
    for deg in degrees:
        if deg in degrees_nb:
            degrees_nb[deg] += 1
    degree_distribution = list(degrees_nb.values())
    for i in range(len(degree_distribution)):
        degree_distribution[i] /= nb_nd
    x = np.arange(0, 5, 0.1)
    fig.add_trace(go.Scatter(x=x, y=degree_distribution, mode='lines', name='Degree Distribution'), row=2, col=2)
    # SLIDER FOR POISSON LAMBDA
    for step in np.arange(0, 1.5, 0.01):
        fig.add_trace(
            go.Scatter(
                visible=False,
                mode='lines',
                name="Poisson: lambda=" + str(round(step, 3)),
                x=x,
                y=np.exp(-step)*np.power(step, x)/factorial(x)), row=2, col=2)
    fig.data[10].visible = True
    steps = []
    for i in range(146):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}]  # layout attribute
        )
        step["args"][0]["visible"][i+4] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][0] = True
        step["args"][0]["visible"][1] = True
        step["args"][0]["visible"][2] = True
        step["args"][0]["visible"][3] = True
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "lambda: "},
        steps=steps
    )]
    fig.update_layout(
        sliders=sliders
    )
    # WRITE HTML FILE:
    if output_path is not None:
        fig.write_html(output_path)
