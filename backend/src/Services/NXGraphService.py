# IMPORTS=
import heapq
import math
import os
import pickle
from collections import OrderedDict
from datetime import datetime, date, timedelta
import networkx as nx
from itertools import permutations
import networkx.algorithms.community as nx_comm
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import cm, colors
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from scipy.special import factorial
import time

# MODELS=
from src.Models.NXGraph import NXGraph
from src.Models.Graph import Graph

# WARNINGS=
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=SyntaxWarning)


# COMMON DATA/SETTINGS FUNCTIONS:
def fig_update_layout(fig):
    """
    This function updates the layout of the figure with specific parameters that are used throughout the project.
    :param fig: The figure to update its layout.
    """
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"))
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'modebar_bgcolor': 'rgba(0, 0, 0, 0)',
        'modebar_color': 'white',
        'margin': dict(l=0, r=0, t=35, b=0),
        'coloraxis_colorbar': dict(titlefont=dict(color="white"), tickfont=dict(color="white")),
        'legend_font_color': 'white',
        'legend_font_size': 16
    })
    fig.update_layout(title_font_color="white",
                      title_font_size=20,
                      title_x=0.5)


def empty_map(pickle_path, title, output_path):
    """
    This function creates an empty map, necessary for the first page of the app when no parameters are selected in the form.
    :param pickle_path: The path of the pickle file.
    :param title: The title of the figure.
    :param output_path: The output path of the html file.
    """
    fig = px.scatter_mapbox(
        center=dict(lat=37, lon=106) if pickle_path == "static/output/chinese.pickle" else dict(lat=21, lon=80),
        zoom=3.4 if pickle_path == "static/output/chinese.pickle" else 4.2,
        mapbox_style="open-street-map", height=740)
    fig_update_layout(fig)
    fig.update_layout(title_text=title)
    # WRITE HTML FILE:
    if output_path is not None:
        fig.write_html(output_path)


def custom_error(my_error):
    """
    This function creates a custom error page to display the raised error to the user. (Metrics not implemented,
    components not implemented, etc.)
    :param my_error: The error message to display.
    """
    file = open("static/output/plotly.html", "w")
    file.write('<div class="row">'
               '<div class="col"></div><div class="col"><br>'
               '<h3 class="text-lg-center" style="color:red">ERROR :(</h3>'
               '<div class="card bg-gradient-danger"><br>'
               '<i class="material-icons-round m-2 align-self-center" style="color:black; font-size: xxx-large">'
               'warning</i><br>'
               '<p class="text-bold align-self-center text-lg-center" id="err" style="color:black">' + my_error +
               '</p>''<br></div></div><div class="col"></div></div><br><br>')
    file.close()


# HELPER FUNCTIONS:
def df_from_nxgraph(nxgraph, component="node"):
    """
    This function creates a dataframe from a networkx graph object, with values needed for the maps.
    The values are different for nodes and edges, so the component parameter is used to distinguish between them.
    :param nxgraph: The networkx graph object.
    :param component: The component to create the dataframe for (either node or edge).
    :return: The dataframe.
    """
    node_metrics_dict = {
        "total_passages": "Total passages",
        "total_minutes": "Total minutes"
    }
    edge_metrics_dict = {
        "mileage": "Mileage",
        "total_travels": "Total travels",
        "total_minutes": "Total minutes",
        "total_mileage": "Total mileage"
    }
    if component == "node":
        nodes = []
        for node in nxgraph.nodes(data=True):
            element = [node[0], node[1]['lat'], node[1]['lon']]
            for metric in node_metrics_dict.keys():
                element.append(node[1][metric])
            nodes.append(element)
        columns = ['Node ID', 'Latitude', 'Longitude'] + list(node_metrics_dict.values())
        df = pd.DataFrame(nodes, columns=columns)
    elif component == "edge":
        edges = []
        for edge in nxgraph.edges(data=True):
            element = [edge[0], edge[1], edge[2]['fromLat'], edge[2]['fromLon'], edge[2]['destLat'], edge[2]['destLon']]
            for metric in edge_metrics_dict.keys():
                element.append(edge[2][metric])
            edges.append(element)
        columns = ['Source Node', 'Destination Node', 'Source Latitude', 'Source Longitude', 'Destination Latitude',
                   'Destination Longitude'] + list(edge_metrics_dict.values())
        df = pd.DataFrame(edges, columns=columns)
    else:
        raise Exception("Component not recognized.")
    return df


def largest_connected_component_ratio(original_graph, attacked_graph):
    """
    This function calculates the ratio between the largest connected component of the attacked graph and the largest
    connected component of the original graph.
    :param original_graph: The original graph object (networkx).
    :param attacked_graph: The attacked graph object (networkx).
    :return: The ratio between the largest connected component of the attacked graph and the largest connected component
    of the original graph.
    """
    og_cc, cc = nx.strongly_connected_components(original_graph), nx.strongly_connected_components(attacked_graph)
    og_lcc, lcc = max(og_cc, key=len), max(cc, key=len)

    return len(lcc) / len(og_lcc)


def global_efficiency_weighted(graph, weight='mileage'):
    """
    This function calculates the global efficiency of a graph (networkx) with weighted edges (default: mileage).
    It is calculated as the average of the inverse of the shortest paths between all pairs of nodes.
    :param graph: The graph object (networkx).
    :param weight: The weight of the edges to consider (default: mileage).
    :return: The global efficiency of the graph.
    """
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
    """
    This function calculates the ratio between the global efficiency of the attacked graph and the global efficiency
    of the original graph.
    :param original_graph: The original graph object (networkx).
    :param attacked_graph: The attacked graph object (networkx).
    :return: The ratio between the global efficiency of the attacked graph and the global efficiency of the original
    """
    return global_efficiency_weighted(attacked_graph) / global_efficiency_weighted(original_graph)


def show_cluster_info(nx_graph, clusters, fig, weight, adv_legend):
    """
    This function shows the information about the clusters in the graph.
    :param nx_graph: The graph object (networkx).
    :param clusters: The clusters of the graph.
    :param fig: The figure to plot on.
    :param weight: The weight of the edges to consider (default: mileage).
    :param adv_legend: Whether to show the advanced legend or not (default: False).
    """
    avg_degrees, avg_bet, avg_load = {}, {}, {}
    avg_deg, avg_between, avg_ld = 0, 0, 0

    if adv_legend:
        degree = nx.degree_centrality(nx_graph)
        between = nx.betweenness_centrality(nx_graph, weight=weight)
        load = nx.load_centrality(nx_graph, weight=weight)
    i = 0
    for cluster in clusters:
        i += 1
        if adv_legend:
            for node in cluster:
                avg_deg += degree[node]
                avg_between += between[node]
                avg_ld += load[node]

            avg_deg /= len(cluster)
            avg_between /= len(cluster)
            avg_ld /= len(cluster)

            avg_degrees[i] = avg_deg
            avg_bet[i] = avg_between
            avg_load[i] = avg_ld

        # Create a scatter trace for the cluster
        cluster_lat = [np.round(nx_graph.nodes[node]['lat'], 2) for node in cluster]
        cluster_lon = [np.round(nx_graph.nodes[node]['lon'], 2) for node in cluster]
        cluster_text = [f'Node n°{node}<br>Cluster n°{i}<br>' for node in cluster]

        fig.add_trace(
            go.Scattermapbox(
                lat=cluster_lat,
                lon=cluster_lon,
                mode='markers',
                marker=dict(size=5, color=i),
                name=(f'Cluster {i}<br>'
                      f'<span style="font-size: 10px;">'
                      f'Betweenness centrality: {avg_bet[i]:.4f}<br>'
                      f'Degree centrality: {avg_degrees[i]:.4f}<br>'
                      f'Load centrality: {avg_load[i]:.4f}'
                      f'</span>') if adv_legend else (
                    f'Cluster {i}<br>'
                    f'<span style="font-size: 10px;">'
                    f'</span>'),
                hoverinfo='text',
                hovertext=cluster_text
            )
        )

    # Create custom legend with metrics for each cluster
    legend_items = []
    for i in range(1, len(clusters) + 1):
        legend_title = f'Cluster {i}'
        metrics_text = f'Cluster {i}<br>' \
                       f'Average degree centrality: {avg_degrees[i]:.6f}<br>' \
                       f'Average betweenness centrality: {avg_bet[i]:.6f}<br>' if adv_legend else f'Cluster {i}<br>'

        legend_items.append(dict(label=metrics_text, method='update', args=[
            {'title': metrics_text, 'showlegend': True, 'legend_title': legend_title,
             'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': -0.25, 'xanchor': 'left', 'x': 0}}
        ]))


# DEFAULT PLOTTING OF NODES:
def plotly_default(pickle_path, day=None, output_path=None):
    """
    This function plots the nodes of the graph on a map with the default attributes (total passages and total minutes).
    :param pickle_path: The path of the pickle file containing the graph.
    :param day: The day to consider (default: None).
    :param output_path: The path of the output file (default: None) in which to save the .html file.
    :return: The figure.
    """
    nxgraph = NXGraph(pickle_path=pickle_path, dataset_number=1,
                      day=int(day) if day is not None and day != "" else None)
    # NODES DATAFRAME:
    df_nodes = df_from_nxgraph(nxgraph, component="node")
    # EDGES DATAFRAME:
    df_edges = df_from_nxgraph(nxgraph, component="edge")
    # CREATE NEW FIGURE:
    fig = px.scatter_mapbox(df_nodes[df_nodes["Node ID"] == -1], lat="Latitude", lon="Longitude", hover_name="Node ID",
                            hover_data=["Total passages", "Total minutes"],
                            zoom=3.4 if pickle_path == "static/output/chinese.pickle" else 4.2,
                            mapbox_style="open-street-map", height=740,
                            center=dict(lat=37, lon=106) if pickle_path == "static/output/chinese.pickle" else dict(
                                lat=21, lon=80))
    # (lat=36, lon=117)
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
            str) + "<br>Total travels: " + df_edges['Total travels'].astype(str) + "<br>Total minutes: " + df_edges[
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
        name="Nodes"
    )
    # GLOBAL SETTINGS:
    fig_update_layout(fig)
    # TITLE:
    fig.update_layout(title_text=f"Default Plotly Window " + (
        "(Day " + str(day) + ", " if day is not None and day != "" else "(") + str(
        len(nxgraph.nodes())) + " nodes and " + str(len(nxgraph.edges())) + " edges)")
    # OUTPUT:
    if output_path is not None:
        fig.write_html(output_path)
    return fig


# PLOTLY HEATMAP:
def plotly_heatmap(pickle_path, component=None, metric=None, day=None, output_path=None):
    """
    This function plots the heatmap of the graph on a map with the specified parameters (component, metric, day).
    :param pickle_path: The path of the pickle file containing the graph.
    :param component: The component to consider (node or edge).
    :param metric: The metric to consider (total_minutes, total_passages, degree_centrality, betweenness_centrality).
    :param day: The day to consider (default: None).
    :param output_path: The path of the output file (default: None) in which to save the .html file.
    :return: The figure.
    """
    if component is None:
        return empty_map(pickle_path, "Heatmap", output_path)
    # NXGRAPH:
    nxgraph = NXGraph(pickle_path=pickle_path, dataset_number=1,
                      day=int(day) if day is not None and day != "" else None)
    # SETTINGS:
    var_factor = 3
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
            return custom_error("Metric not implemented!")
        # PLOTTING SETTINGS:
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
                                center=dict(lat=37, lon=106) if pickle_path == "static/output/chinese.pickle" else dict(
                                    lat=21, lon=80),
                                zoom=3.4 if pickle_path == "static/output/chinese.pickle" else 4.2,
                                mapbox_style="open-street-map", height=740,
                                range_color=[vmin, vmax], hover_name='Node ID',
                                hover_data=['Latitude', 'Longitude', metric_name])
    elif component == "edge":
        if metric == "total_minutes":
            metric_name = "Total minutes"
        elif metric == "total_passages":
            metric = "total_travels"
            metric_name = "Total travels"
        elif metric == "mileage":
            metric_name = "Mileage"
        elif metric == "total_mileage":
            metric_name = "Total mileage"
        elif metric == "degree_centrality":
            return custom_error("Degree centrality not implemented for edges!")
        elif metric == "betweenness_centrality":
            metric_name = "Betweenness centrality"
            print("Gathering betweenness centralities...")
            betweenness_centrality = nx.edge_betweenness_centrality(nxgraph)
            nx.set_edge_attributes(nxgraph, betweenness_centrality, 'betweenness_centrality')
        elif metric == "closeness_centrality":
            return custom_error("Closeness centrality not implemented for edges!")
        else:
            return custom_error("Metric not implemented")
        # DUMMY PLOT:
        df_nodes = df_from_nxgraph(nxgraph, component="node")
        fig = px.scatter_mapbox(df_nodes[df_nodes["Node ID"] == -1], lat="Latitude", lon="Longitude",
                                hover_name="Node ID",
                                hover_data=["Total passages", "Total minutes"],
                                zoom=3.4 if pickle_path == "static/output/chinese.pickle" else 4.2,
                                mapbox_style="open-street-map",
                                height=740,
                                center=dict(lat=37, lon=106) if pickle_path == "static/output/chinese.pickle" else dict(
                                    lat=21, lon=80))
        # PLOTTING SETTINGS:
        min_metric = min([edge[2][metric] for edge in nxgraph.edges(data=True)])
        max_metric = max([edge[2][metric] for edge in nxgraph.edges(data=True)])
        median_metric = np.median([edge[2][metric] for edge in nxgraph.edges(data=True)])
        avg_metric = np.mean([edge[2][metric] for edge in nxgraph.edges(data=True)])
        metrics = sorted([edge[2][metric] for edge in nxgraph.edges(data=True)])
        vmin = min_metric
        vmax = max_metric
        # Steps array using the median metric so that the color scale is centered on it:
        steps = [metrics[int(len(metrics) * i / 10)] for i in range(10)]
        # Colors from dark blue to purple, to pink, to orange, to yellow:
        colors = [
            "rgb(18, 8, 137)",
            "rgb(70, 3, 159)",
            "rgb(114, 1, 168)",
            "rgb(153, 22, 159)",
            "rgb(196, 63, 127)",
            "rgb(223, 98, 100)",
            "rgb(244, 140, 70)",
            "rgb(252, 185, 46)",
            "rgb(241, 244, 33)",
        ]
        # EDGES: Add edges as disconnected lines in a single trace
        edges_x = [[] for i in range(len(steps))]
        edges_y = [[] for i in range(len(steps))]
        for edge in nxgraph.edges(data=True):
            x0, y0 = nxgraph.nodes[edge[0]]['lon'], nxgraph.nodes[edge[0]]['lat']
            x1, y1 = nxgraph.nodes[edge[1]]['lon'], nxgraph.nodes[edge[1]]['lat']
            # Find the range step of the edge metric:
            for i in range(len(steps) - 1):
                step = 0
                if steps[i] <= edge[2][metric] and edge[2][metric] <= steps[i + 1]:
                    step = i
                    break
            edges_x[step].append(x0)
            edges_x[step].append(x1)
            edges_x[step].append(None)
            edges_y[step].append(y0)
            edges_y[step].append(y1)
            edges_y[step].append(None)
        for i in range(len(steps) - 1):
            fig.add_trace(go.Scattermapbox(
                lat=edges_y[i],
                lon=edges_x[i],
                mode='lines',
                line=dict(width=1, color=colors[i]),
                hoverinfo='text',
                showlegend=True,
                name=metric_name + "(" + str(format(steps[i], '.1E')) + " > " + str(format(steps[i + 1], '.1E')) + ")",
            ))
    # GLOBAL SETTINGS:
    fig_update_layout(fig)
    fig.update_layout(title_text=f"Heatmap: Component=" + component + ", Metric=" + metric_name
                                 + (f", Day={day}" if day is not None and day != "" else ""))
    # WRITE HTML FILE:
    if output_path is not None:
        fig.write_html(output_path)
    return fig


# PLOTLY RESILIENCE: TODO: add edges (red/white) + add sub functions for duplicated code
def plotly_resilience(pickle_path, day=None, strategy=None, component=None, metric=None, fraction=None,
                      output_path=None, smallworld=None):
    """
    This function plots the resilience of a network (ie. the nodes or edges that need to be removed to disconnect it).
    It is given a pickle file and a strategy (random or targeted) and a component (node or edge). It is also given a
    metric (degree_centrality, betweenness_centrality, closeness_centrality) and a fraction (0.1, 0.2, 0.3, 0.4, 0.5).
    It returns a plotly map that highlights the nodes or edges that need to be removed to disconnect the network.
    :param pickle_path: The path to the pickle file of the Graph object.
    :param day: The day of the network to plot.
    :param strategy: The strategy to use (random or targeted).
    :param component: The component to remove (node or edge).
    :param metric: The metric to use (degree_centrality, betweenness_centrality, closeness_centrality, etc).
    :param fraction: The fraction of nodes or edges to remove.
    :param output_path: The path to the output html file.
    :param smallworld: The smallworldness of the network.
    :return: A plotly map that highlights the nodes or edges that need to be removed to disconnect the network.
    """
    if component is None:
        return empty_map(pickle_path, "Resilience", output_path)
    nxgraph = NXGraph(pickle_path=pickle_path, dataset_number=1,
                      day=int(day) if day is not None and day != "" else None)
    nx_graph_copy = nxgraph.copy()
    lcc, global_eff = 0, 0
    # FRACTION VERIFICATION:
    if fraction is None or fraction == "":
        return custom_error("Fraction not specified!")
    else:
        fraction = float(fraction)
    if strategy == "targeted":
        if component == "node":
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
            return custom_error("Targetted edge resilience not implemented yet...")
    elif strategy == "random":
        if component == "node":
            return custom_error("Random node resilience not implemented yet...")
        elif component == "edge":
            return custom_error("Random edge resilience not implemented yet...")

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
                            center=dict(lat=37, lon=106) if pickle_path == "static/output/chinese.pickle" else dict(
                                lat=21, lon=80),
                            zoom=3.4 if pickle_path == "static/output/chinese.pickle" else 4.2,
                            mapbox_style="open-street-map", height=740)
    fig.add_scattermapbox(lat=init_df['Latitude'], lon=init_df['Longitude'], mode='markers',
                          marker=dict(size=3, color='blue'),
                          name="Nodes", hoverinfo="text",
                          hovertext="Node n°" + init_df['Node ID'].astype(str) + "<br>" + metric + ": " + init_df[
                              metric].astype(str))
    fig.add_scattermapbox(lat=destroyed_df['Latitude'], lon=destroyed_df['Longitude'], mode='markers',
                          marker=dict(size=12, color='red'),
                          name="Destroyed nodes", hoverinfo="text",
                          hovertext="Node n°" + destroyed_df['Node ID'].astype(str) + "<br>" + metric + ": " +
                                    destroyed_df[metric].astype(str))
    fig.add_scatter(x=[None], y=[None], mode='none', name='LCC size ratio: ' + str(round(lcc,3))
                                                  + '<br>Global efficiency ratio: ' + str(round(global_eff, 3)))
    # GLOBAL SETTINGS:
    fig_update_layout(fig)
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"))
    fig.update_layout(
        title_text=f"Resilience: Strategy={strategy}, Component={component}, Metric={metric}, Fraction={fraction}"
                   + (f"Day={day}" if day is not None and day != "" else ""),
        title_font_color="white",
        title_font_size=20,
        title_x=0.5)
    # WRITE HTML FILE:
    if output_path is not None:
        fig.write_html(output_path)
    if smallworld == "true":
        plotly_small_world(pickle_path, day, output_path, nx_graph_copy)
    return fig


# PLOTLY CLUSTERING:
def plotly_clustering(pickle_path, day=None, algorithm=None, weight=None, output_path=None, adv_legend=False):
    """
    This function plots the clustering of a graph using the algorithm specified in the parameters and saves it in the
    output_path if specified.
    :param pickle_path: The path to the pickle file containing the graph
    :param day: The day to consider
    :param algorithm: The algorithm to use
    :param weight: The weight to use
    :param output_path: The path to the output file
    :param adv_legend: Whether to use the advanced legend or not
    :return: The plotly figure.
    """
    if algorithm is None:
        return empty_map(pickle_path, "Clustering", output_path)
    nxgraph = NXGraph(pickle_path=pickle_path, dataset_number=1,
                      day=int(day) if day is not None and day != "" else None)
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
        communities = nx_comm.louvain_communities(nxgraph, weight=weight)

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
                            center=dict(lat=37, lon=106) if pickle_path == "static/output/chinese.pickle" else dict(
                                lat=21, lon=80),
                            zoom=3.4 if pickle_path == "static/output/chinese.pickle" else 4.2,
                            mapbox_style="open-street-map", height=740)

    show_cluster_info(nxgraph, communities, fig, weight, adv_legend=adv_legend)
    # GLOBAL SETTINGS:
    fig_update_layout(fig)
    clust_title = f"Clustering: Algorithm={algorithm}"
    if day is not None and day != "":
        clust_title += f", Day={day}"
    if algorithm == "Louvain":
        clust_title += f", Weight={weight}"
    fig.update_layout(title_text=clust_title)
    # WRITE HTML FILE:
    if output_path is not None:
        fig.write_html(output_path)
    return fig


# PLOTLY SMALL WORLD:
def plotly_small_world(pickle_path, day=None, output_path=None, nxgraph=None):
    """
    This function plots the small world features of a graph and saves it in the output_path if specified.
    Among the features it offers, there are the shortest path lenghts, the clustering coefficients, the degrees of each
    node and the degree distribution with the power law and Poisson distributions.
    :param pickle_path: The path to the pickle file containing the graph
    :param day: The day to consider
    :param output_path: The path to the output file
    :param nxgraph: The NetworkX Graph to use
    :return: The plotly figure.
    """
    complement = True
    if nxgraph is None:
        nxgraph = NXGraph(pickle_path=pickle_path, dataset_number=1,
                          day=int(day) if day is not None and day != "" else None)
        complement = False
    fig = make_subplots(rows=2, cols=2)
    fig.update_layout(title_text=f"Small-World Features " + (
        "(day " + str(day) + ")" if day is not None and day != "" else ""), height=700)
    nbins = 100
    nb_nd = nxgraph.number_of_nodes()

    # SHORTEST PATH HISTOGRAM -> small diameter
    timer = time.time()
    shortest = dict(nx.shortest_path_length(nxgraph))
    print("All shortest paths computed in {} seconds.".format(round(time.time() - timer, 2)))
    path_numbers = {}
    for start in shortest.keys():
        from_start = dict(shortest[start])
        for arrival in from_start.keys():
            if from_start[arrival] not in path_numbers:
                path_numbers.update({from_start[arrival]: 1})
            else:
                path_numbers[from_start[arrival]] += 1
    fig.add_trace(go.Scatter(x=list(path_numbers.keys()), y=list(path_numbers.values()), mode='lines+markers',
                             name="Shortest path length",
                             hovertemplate='<b>Shortest Path Length Histogram</b><br>Shortest path length: %{x}'
                                           '<br>Number of paths: %{y}<br><extra></extra>'), row=1, col=1)
    total_dist = 0
    nb_paths = 0
    for i in path_numbers.keys():
        total_dist += i * path_numbers[i]
        nb_paths += path_numbers[i]
    path_length_mean = total_dist / nb_paths
    annot_length = "Mean = " + str(round(path_length_mean, 2))
    fig.add_vline(x=path_length_mean, line_dash="dot", annotation_text=annot_length, annotation_position="top right",
                  row=1, col=1)
    fig.update_yaxes(title_text="Number of paths", row=1, col=1, title_standoff=0)
    fig.update_xaxes(title_text="Shortest path length", row=1, col=1, title_standoff=0)

    # DEGREE HISTOGRAM -> some high degree nodes
    degrees = list(dict(nx.degree(nxgraph)).values())
    fig.add_trace(go.Histogram(x=degrees, nbinsx=nbins, name="Degree",
                               hovertemplate='<b>Degree Histogram</b><br>Degree: %{x}'
                                             '<br>Number of nodes: %{y}<br><extra></extra>'), row=1, col=2)
    fig.update_yaxes(title_text="Number of nodes", row=1, col=2, title_standoff=0)
    fig.update_xaxes(title_text="Degree", row=1, col=2, title_standoff=0)

    # CLUSTERING COEFFICIENT HISTOGRAM -> highly clustered
    graph = nx.Graph(nxgraph)  # nx.clustering not implemented for multigraphs
    clustering_coeffs = list(dict(nx.clustering(graph)).values())
    fig.add_trace(go.Histogram(x=clustering_coeffs, nbinsx=nbins, name="Clustering coefficients",
                               hovertemplate='<b>Clustering Coefficient Histogram</b><br>Clustering coefficient: %{x}'
                                             '<br>Number of nodes: %{y}<br><extra></extra>'), row=2, col=1)
    clust_coeff_mean = sum(clustering_coeffs) / len(clustering_coeffs)
    annot_clust = "Network CC = " + str(round(clust_coeff_mean, 2))
    fig.add_vline(x=clust_coeff_mean, line_dash="dot", annotation_text=annot_clust, annotation_position="top right",
                  row=2, col=1)
    fig.update_yaxes(title_text="Number of nodes", row=2, col=1, title_standoff=0)
    fig.update_xaxes(title_text="Clustering Coefficient", row=2, col=1, title_standoff=0)

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
    x = np.arange(0, 30, 1)
    fig.add_trace(go.Scatter(x=x, y=degree_distribution, mode='lines', name='Degree distribution',
                             hovertemplate='<b>Degree Distribution</b><br><i>x=degree</i><br>x: %{x}'
                                           '<br>P(x): %{y}<br><extra></extra>'), row=2, col=2)
    fig.update_yaxes(title_text="P(x)", row=2, col=2, title_standoff=0, range=[0, 0.6])
    fig.update_xaxes(title_text="x", row=2, col=2, title_standoff=0)

    # SLIDER FOR POISSON LAMBDA
    for step in np.arange(0, 10, 0.1):
        fig.add_trace(
            go.Scatter(
                visible=False,
                mode='lines',
                name="Poisson: λ=" + str(round(step, 3)),
                x=x,
                y=np.exp(-step) * np.power(step, x) / factorial(x),
                hovertemplate='<b>Poisson Distribution</b>'
                              '<br><i>P(x)=exp(-λ).λ^x/x!</i>'
                              '<br>x: %{x}'
                              '<br>P(x): %{y}<extra></extra>'
            ), row=2, col=2)
    for step in np.arange(0, 10, 0.1):
        fig.add_trace(
            go.Scatter(
                visible=False,
                mode='lines',
                name="Power law: λ=" + str(round(step, 3)),
                x=x,
                y=np.power(x, -step),
                hovertemplate='<b>Power Law Distribution</b>'
                              '<br><i>P(x)=x^-λ</i>'
                              '<br>x: %{x}'
                              '<br>P(x): %{y}<extra></extra>'
            ), row=2, col=2)
    fig.data[14].visible = True
    fig.data[114].visible = True
    steps = []
    for i in range(96):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}]  # layout attribute
        )
        step["args"][0]["visible"][i + 4] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][i + 104] = True  # Toggle i'th trace to "visible"
        for i in range(4):
            step["args"][0]["visible"][i] = True
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "λ: "},
        steps=steps
    )]
    fig.update_layout(
        sliders=sliders
    )

    # WRITE HTML FILE:
    if output_path is not None and not complement:
        fig.write_html(output_path)
    else:
        fig.write_html("static/output/temp.html")
        map_plot = open(output_path, "a")
        graph_plot = open("static/output/temp.html", "r", errors="ignore")
        map_plot.write(graph_plot.read())
        map_plot.close()
        graph_plot.close()
        os.remove("static/output/temp.html")
    return fig


# PLOTLY CENTRALITY:
def plotly_centrality(pickle_path, day=None, output_path=None):
    """
    This function plots the centrality measures of the graph.
    :param pickle_path: The path to the pickle file containing the graph.
    :param day: The day to plot the centrality measures for.
    :param output_path: The path to the output file.
    :return: The centrality plot figure.
    """
    nxgraph = NXGraph(pickle_path=pickle_path, dataset_number=1,
                          day=int(day) if day is not None and day != "" else None)
    fig = make_subplots(rows=3, cols=1)
    fig.update_layout(title_text=f"Centrality Measures " + (
        "(day " + str(day) + ")" if day is not None and day != "" else ""), height=700)
    nbins = 100
    nb_nd = nxgraph.number_of_nodes()

    # DEGREE CENTRALITY HISTOGRAM
    deg_c = list(dict(nx.degree_centrality(nxgraph)).values())
    fig.add_trace(go.Histogram(x=deg_c, nbinsx=nbins, name="Degree Centrality",
                               hovertemplate='<b>Degree Centrality Histogram</b><br>Degree centrality: %{x}'
                                             '<br>Number of nodes: %{y}<br><extra></extra>'), col=1, row=1)
    fig.update_yaxes(title_text="Number of nodes", row=1, col=1, title_standoff=0)
    fig.update_xaxes(title_text="Degree Centrality", row=1, col=1, title_standoff=0)
    avg_deg_c = sum(deg_c) / len(deg_c)
    annot_deg = "avg = " + str(round(avg_deg_c, 5))
    fig.add_vline(x=avg_deg_c, line_dash="dot", annotation_text=annot_deg, annotation_position="top right",
                  row=1, col=1,)

    # BETWEENNESS CENTRALITY HISTOGRAM
    bet_c = list(dict(nx.betweenness_centrality(nxgraph)).values())
    fig.add_trace(go.Histogram(x=bet_c, nbinsx=nbins, name="Betweenness Centrality",
                               hovertemplate='<b>Betweenness Centrality Histogram</b><br>Betweenness centrality: %{x}'
                                             '<br>Number of nodes: %{y}<br><extra></extra>'), col=1, row=2)
    fig.update_yaxes(title_text="Number of nodes", col=1, row=2, title_standoff=0)
    fig.update_xaxes(title_text="Betweenness Centrality", col=1, row=2, title_standoff=0)
    avg_bet_c = sum(bet_c) / len(bet_c)
    annot_bet = "avg = " + str(round(avg_bet_c, 5))
    fig.add_vline(x=avg_bet_c, line_dash="dot", annotation_text=annot_bet, annotation_position="top right",
                  col=1, row=2)

    # CLOSENESS CENTRALITY HISTOGRAM
    clos_c = list(dict(nx.closeness_centrality(nxgraph)).values())
    fig.add_trace(go.Histogram(x=clos_c, nbinsx=nbins, name="Closeness Centrality",
                               hovertemplate='<b>Closeness Centrality Histogram</b><br>Closeness centrality: %{x}'
                                             '<br>Number of nodes: %{y}<br><extra></extra>'), col=1, row=3)
    fig.update_yaxes(title_text="Number of nodes", col=1, row=3, title_standoff=0)
    fig.update_xaxes(title_text="Closeness Centrality", col=1, row=3, title_standoff=0)
    avg_clos_c = sum(clos_c) / len(clos_c)
    annot_clos = "avg = " + str(round(avg_clos_c, 5))
    fig.add_vline(x=avg_clos_c, line_dash="dot", annotation_text=annot_clos, annotation_position="top right",
                  col=1, row=3)

    # WRITE HTML FILE:
    if output_path is not None:
        fig.write_html(output_path)
    return fig


# PLOTLY CORRELATION:
def plotly_correlation(pickle_path, day=None, output_path=None, neighbor=None):
    nxgraph = NXGraph(pickle_path=pickle_path, dataset_number=1,
                          day=int(day) if day is not None and day != "" else None)
    fig = make_subplots(rows=2, cols=1)
    if neighbor == 'true':
        fig = make_subplots(rows=3, cols=1)
    fig.update_layout(title_text=f"Correlation Measures " + (
        "(day " + str(day) + ")" if day is not None and day != "" else ""), height=700)
    nbins = 100
    nb_nd = nxgraph.number_of_nodes()

    # DEGREE CORRELATION, CLUSTERING DEGREE CORRELATION
    avg_nbh_degrees = dict(nx.average_neighbor_degree(nxgraph))
    degrees = dict(nx.degree(nxgraph))
    clus = dict(nx.clustering(nx.Graph(nxgraph)))
    nb_deg = {}
    sum_avnede_deg = {}
    sum_clus_deg = {}
    for node in degrees:
        if degrees[node] not in nb_deg:
            nb_deg.update({degrees[node]: 1})
        else:
            nb_deg[degrees[node]] += 1
        if degrees[node] not in sum_avnede_deg:
            sum_avnede_deg.update({degrees[node]: avg_nbh_degrees[node]})
        else:
            sum_avnede_deg[degrees[node]] += avg_nbh_degrees[node]
        if degrees[node] not in sum_clus_deg:
            sum_clus_deg.update({degrees[node]: clus[node]})
        else:
            sum_clus_deg[degrees[node]] += clus[node]
    deg_corr = []
    for deg in nb_deg:
        deg_corr.append(sum_avnede_deg[deg]/nb_deg[deg])
    fig.add_trace(go.Scatter(x=list(nb_deg.keys()),
                             y=deg_corr,
                             mode='markers',
                             name="Degree Correlation",
                             hovertemplate='<b>Degree Correlation</b><br>Degree: %{x}'
                                           '<br>Degree correlation: %{y}<br><extra></extra>'), row=1, col=1)
    fig.update_yaxes(title_text="Degree Correlation", row=1, col=1, title_standoff=0)
    fig.update_xaxes(title_text="Degree", row=1, col=1, title_standoff=0)
    clus_deg_corr = []
    for deg in nb_deg:
        clus_deg_corr.append(sum_clus_deg[deg] / nb_deg[deg])
    fig.add_trace(go.Scatter(x=list(nb_deg.keys()),
                             y=clus_deg_corr,
                             mode='markers',
                             name="Clustering-Degree Correlation",
                             hovertemplate='<b>Clustering-Degree Correlation</b><br>Degree: %{x}'
                                           '<br>Clustering-degree correlation: %{y}<br><extra></extra>'), row=2, col=1)
    fig.update_yaxes(title_text="Clustering-Degree Correlation", row=2, col=1, title_standoff=0)
    fig.update_xaxes(title_text="Degree", row=2, col=1, title_standoff=0)

    # AVERAGE NEIGHBOR DEGREE HISTOGRAM
    if neighbor == 'true':
        list_avnede = list(avg_nbh_degrees.values())
        fig.add_trace(go.Histogram(x=list_avnede, nbinsx=nbins, name="Average Neighbor Degree",
                                   hovertemplate='<b>Average Neighbor Degree Histogram</b><br>Average neighbor degree: %{x}'
                                                 '<br>Number of nodes: %{y}<br><extra></extra>'), col=1, row=3)
        fig.update_yaxes(title_text="Number of nodes", row=3, col=1, title_standoff=0)
        fig.update_xaxes(title_text="Average Neighbor Degree", row=3, col=1, title_standoff=0)
        avg_avg_nede = sum(list_avnede) / len(list_avnede)
        annot = "avg = " + str(round(avg_avg_nede, 5))
        fig.add_vline(x=avg_avg_nede, line_dash="dot", annotation_text=annot, annotation_position="top right",
                      row=3, col=1, )

    # WRITE HTML FILE:
    if output_path is not None:
        fig.write_html(output_path)
    return fig


# SHORTEST PATH ANALYSIS
def astar_path(graph, start, end):
    """
    Finds the shortest path between the start and end nodes of the given graph using the A* algorithm.
    :param graph: The graph to search.
    :param start: The starting node.
    :param end: The ending node.
    :return: A list of nodes representing the shortest path.
    """
    path = nx.astar_path(graph, start, end, heuristic=lambda u, v: heuristic_function(graph, u, v), weight='mileage')
    return path


def heuristic_function(nx_graph, u, v):
    """
    Heuristic function to estimate the distance between the current node and the target node.
    The function return the inverse of the distance as nodes with higher distances are the worst candidates.
    :param nx_graph: railway network
    :param u: The current node.
    :param v: The target node.
    :return: An estimate of the distance between the two nodes.
    """
    if u == v:
        return 0
    fromLat, fromLon = nx_graph.nodes[u]['lat'], nx_graph.nodes[u]['lon']
    destLat, destLon = nx_graph.nodes[v]['lat'], nx_graph.nodes[v]['lon']

    return 1 / haversine(fromLat, fromLon, destLat, destLon)


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the distance between two points on the Earth's surface using the Haversine formula.
    :param lat1: The latitude of the first point.
    :param lon1: The longitude of the first point.
    :param lat2: The latitude of the second point.
    :param lon2: The longitude of the second point.
    :return: The distance between the two points, in kilometers.
    """
    earth_radius = 6371  # kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c
    return distance


def travel_cost(graph: Graph, current_node, next_node, current_time):
    """
    Calculates the cost of a travel between two nodes.
    :param graph: The NetworkX graph to search on.
    :param current_node: The from node.
    :param next_node: The destination node.
    :param current_time: The current time.
    :return: The cost of the travel.
    """
    best_cost = float('inf')
    best_travel = 0
    best_arrival = 0
    # for each edge between current_node and next_node (max is 2)
    for rail in graph.get_edges()[current_node, next_node]:
        # for each travel on the edge
        for travel in rail.get_travels():
            dep_time = datetime.strptime(travel.get_dep_time(), '%H:%M:%S').time()

            if dep_time < current_time:
                dt1 = datetime.combine(date.today(), current_time)
                dt2 = datetime.combine(date.today() + timedelta(days=1), dep_time)
            else:
                dt1 = datetime.combine(date.today(), current_time)
                dt2 = datetime.combine(date.today(), dep_time)

            wait = dt2 - dt1

            cost = round(wait.total_seconds() / 60) + travel.get_travel_time()
            if cost < best_cost:
                best_cost = cost
                best_travel = travel
                best_arrival = dt2 + timedelta(minutes=travel.get_travel_time())

    return best_cost, best_arrival.time(), best_travel.get_train_id()


def a_star_shortest_path(nx_graph: NXGraph, graph: Graph, root, goal, departure_time):
    """
    Computes the shortest path from a station to another departing at a chosen time with A* search.
    :param nx_graph: Railway network
    :param graph: original graph object
    :param root: starting node
    :param goal: target node
    :param departure_time: time of departure
    :return: path and expected arrival time
    """

    # Define a priority queue setting the priority as: f(n) = g(n) + h(n), where g(n) is the
    # cost of the path so far and h(n) is the heuristic cost to the goal node.
    frontier = [(0, root)]
    # path reconstruction dict
    came_from = {}

    # Define a dictionary 'record' to keep track of the cost of the best path so far for each node.
    cost_record = {root: (0, departure_time, '_')}

    times, trains = [], []

    # while queue is not empty
    while frontier:
        # Get the node with the lowest f(n) value from the queue --> 'current_node'.
        _, current = heapq.heappop(frontier)
        # If goal node is found ==> return
        if current == goal:
            path = [current]
            while current != root:
                current = came_from[current]
                path.append(current)
                times.append(cost_record[current][1])
                trains.append(cost_record[current][2])
            path.reverse()
            times.reverse()
            trains.reverse()
            return path, times, trains
        # for each neighbor compute cost(root, neighbor)= record[current_node] + travel_cost(current_node, neighbor)
        for neighbor in nx_graph.successors(current):
            current_time = cost_record[current][1]
            cost_to_neighbor = travel_cost(graph, current, neighbor, current_time=current_time)
            cost = cost_record[current][0] + cost_to_neighbor[0]
            # print(current, '-->', neighbor, '  ', current_time, '-->', cost_to_neighbor[1])
            # if neighbor is not in 'record' the dictionary that keeps track best costs so far for each node...
            # ... or if cost(root, neighbor) < record[neighbor]
            if neighbor not in cost_record or cost < cost_record[neighbor][0]:
                # Add/update the cost for neighbor in 'record'
                cost_record[neighbor] = cost, cost_to_neighbor[1], cost_to_neighbor[2]
                # compute priority = cost(root, current) + heuristic(neighbor, goal)
                priority = cost + heuristic_function(nx_graph, neighbor, goal)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current
                # Add/update train and time to tracking lists

    # if we exit the while loop then goal has not been reached ==> no existing path
    print("No existing path")
    return None


def plotly_shortest_path(pickle_path, dep_time=None, end=None, output_path=None, day=None, start=None):
    """
    Computes the shortest path from a station to another departing at a chosen time with A* search.
    :param pickle_path: The path to the pickle file of the Graph object.
    :param dep_time: The departure time.
    :param end: The destination station.
    :param output_path: The path to the output html file.
    :param day: The day at which the path is computed.
    :param start: The starting station.
    :return: The html file with the plot of the temporal and spatial shortest paths.
    """
    nx_graph = NXGraph(pickle_path=pickle_path, dataset_number=1,
                       day=int(day) if day is not None and day != "" else None)
    if start is None:
        return empty_map(pickle_path, 'Shortest Path', output_path)

    # if not nx_graph.has_node(start) or not nx_graph.has_node(end):
    #     return custom_error('Invalid Node(s)')

    if dep_time is not None:
        dep_time = datetime.strptime(dep_time + ':00', '%H:%M:%S').time()
    start = start if start is not None else start
    end = end if end is not None else end

    graph = pickle.load(open(pickle_path, "rb"))

    path1 = [] if start is None and end is None else astar_path(nx_graph, start, end)
    path2 = [(), (), ()] if start is None and end is None else a_star_shortest_path(nx_graph, graph, start, end,
                                                                                    dep_time)

    # NODES DATAFRAME:
    df_nodes = df_from_nxgraph(nx_graph, component="node")

    # Create the empty map
    fig = px.scatter_mapbox(df_nodes[df_nodes["Node ID"] == -1], lat="Latitude", lon="Longitude", hover_name="Node ID",
                            hover_data=["Total passages", "Total minutes"], mapbox_style="open-street-map",
                            height=800,
                            center=dict(lat=37, lon=106) if pickle_path == "static/output/chinese.pickle" else dict(lat=21, lon=80),
                            zoom=3.4 if pickle_path == "static/output/chinese.pickle" else 4.2)

    # Add the two paths as separate traces
    for i, path in enumerate([path1]):
        path_edges_x = []
        path_edges_y = []
        for j in range(len(path) - 1):
            source = path[j]
            target = path[j + 1]
            source_data = df_nodes[df_nodes['Node ID'] == source].iloc[0]
            target_data = df_nodes[df_nodes['Node ID'] == target].iloc[0]
            path_edges_x += [source_data['Longitude'], target_data['Longitude'], None]
            path_edges_y += [source_data['Latitude'], target_data['Latitude'], None]
        fig.add_trace(go.Scattermapbox(
            lat=path_edges_y,
            lon=path_edges_x,
            mode='lines',
            line=dict(width=2, color='blue' if i==0 else 'red'),
            hoverinfo='skip',
            name='Shortest by mileage'
        ))

    # Define a dictionary that maps each train to a unique color
    train_map = {train: i for i, train in enumerate(set(path2[2]))}
    colormap = cm.get_cmap('tab20', len(train_map))

    # Create a list of unique trains
    trains = list(OrderedDict.fromkeys(path2[2][1:]))

    # Iterate through the trains
    for train in trains:
        # Find all the edges that belong to the train
        edges = [(i, i + 1) for i, t in enumerate(path2[2][1:]) if t == train]

        # Create a list of coordinates and times for the edges, with None values between each pair of nodes
        lon_list = []
        lat_list = []
        time_list = []

        for i, edge in enumerate(edges):
            node_1 = path2[0][edge[0]]
            node_2 = path2[0][edge[1]]
            lon_1 = df_nodes.loc[df_nodes['Node ID'] == node_1, 'Longitude'].values[0]
            lat_1 = df_nodes.loc[df_nodes['Node ID'] == node_1, 'Latitude'].values[0]
            time_1 = path2[1][edge[0]].strftime('%H:%M:%S')

            lon_2 = df_nodes.loc[df_nodes['Node ID'] == node_2, 'Longitude'].values[0]
            lat_2 = df_nodes.loc[df_nodes['Node ID'] == node_2, 'Latitude'].values[0]
            time_2 = path2[1][edge[1]].strftime('%H:%M:%S')

            lon_list.append(lon_1)
            lat_list.append(lat_1)
            time_list.append(time_1)

            lon_list.append(lon_2)
            lat_list.append(lat_2)
            time_list.append(time_2)

            if i < len(edges) - 1:
                lon_list.append(None)
                lat_list.append(None)
                time_list.append(None)

        # Create a dataframe with the edge data
        df_edges = pd.DataFrame({
            'Longitude': lon_list,
            'Latitude': lat_list,
            'Time': time_list,
        })

        # Create a list of hovertext for each line segment
        hovertext = []
        for i in range(len(time_list) - 1):
            if time_list[i] is not None and time_list[i + 1] is not None:
                hovertext.append(
                    'Train ' + str(train) + '<br>' + 'Departure: ' + str(time_list[i]) + '<br>' + 'Arrival: ' + str(
                        time_list[i + 1]))
            else:
                hovertext.append(None)

        # Add the trace to the plot
        color = colors.rgb2hex(colormap(train_map[train]))
        fig.add_trace(go.Scattermapbox(
            mode='lines+markers',
            lon=df_edges['Longitude'],
            lat=df_edges['Latitude'],
            marker={'size': 10, 'color': color},
            line={'width': 2, 'color': color},
            hovertext=hovertext,
            name='Train ' + str(train)
        ))

    # Update the layout
    fig_update_layout(fig)
    # WRITE HTML FILE:
    if output_path is not None:
        fig.write_html(output_path)
    return fig
