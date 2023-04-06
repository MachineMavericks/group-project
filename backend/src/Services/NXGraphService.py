# IMPORTS=
import networkx as nx
import plotly.express as px
import pandas as pd
import numpy as np

import plotly.express as px
import pandas as pd
import numpy as np

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
    # NODES:
    nodes = []
    for node in nxgraph.nodes(data=True):
        id = node[0]
        lat = node[1]['lat']
        lon = node[1]['lon']
        total_passages = node[1]['total_passages']
        total_minutes = node[1]['total_minutes']
        nodes.append([id, lat, lon, total_passages, total_minutes])
    df_nodes = pd.DataFrame(nodes, columns=['Node ID', 'Latitude', 'Longitude', 'Total passages', 'Total minutes'])
    fig = px.scatter_mapbox(df_nodes, lat='Latitude', lon='Longitude', hover_name='Node ID',
                            hover_data=['Latitude', 'Longitude', 'Total passages', 'Total minutes'], zoom=3.5,
                            mapbox_style="open-street-map", height=800)
    fig_update_layout(fig)
    # EDGES:
    # edges = []
    # for edge in nxgraph.edges(data=True):
    #     sourceLat = df_nodes.loc[df_nodes['Node ID'] == edge[0]]['Latitude'].values[0]
    #     sourceLon = df_nodes.loc[df_nodes['Node ID'] == edge[0]]['Longitude'].values[0]
    #     targetLat = df_nodes.loc[df_nodes['Node ID'] == edge[1]]['Latitude'].values[0]
    #     targetLon = df_nodes.loc[df_nodes['Node ID'] == edge[1]]['Longitude'].values[0]
    #     mileage = edge[2]['mileage']
    #     total_travels = edge[2]['total_travels']
    #     total_minutes = edge[2]['total_minutes']
    #     total_mileage= edge[2]['total_mileage']
    #     edges.append([sourceLat, sourceLon, targetLat, targetLon, mileage, total_travels, total_minutes, total_mileage])
    #     fig.add_trace(
    #         px.scatter_geo(
    #             lat=[sourceLat, targetLat],
    #             lon=[sourceLon, targetLon],
    #             hover_name=[edge[0], edge[1]],
    #         ).data[0]
    #     )
    # df_edges = pd.DataFrame(edges, columns=['SourceLat', 'SourceLon', 'TargetLat', 'TargetLon', 'Mileage', 'Total travels', 'Total minutes', 'Total mileage'])
    # fig.update_traces(mode='lines', hovertemplate=None)
    # fig.update_layout(showlegend=False)
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
