# IMPORTS=
import networkx as nx
import plotly.express as px
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
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
    nxgraph = NXGraph(pickle_path=pickle_path, dataset_number=1, day=int(day) if day is not None and day != "" else None)
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
    df_edges = pd.DataFrame(edges, columns=['Source Node', 'Destination Node', 'Source Latitude', 'Source Longitude', 'Destination Latitude', 'Destination Longitude', 'Total travels', 'Total minutes'])
    # CREATE NEW FIGURE:
    fig = px.scatter_mapbox(df_nodes[df_nodes["Node ID"] == -1], lat="Latitude", lon="Longitude", hover_name="Node ID", hover_data=["Total passages", "Total minutes"], zoom=3.5, mapbox_style="open-street-map", height=800, center=dict(lat=36, lon=117))
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
        hovertext= "Edge from "+df_edges['Source Node'].astype(str)+" to "+df_edges['Destination Node'].astype(str)+"<br>Travels: "+df_edges['Total travels'].astype(str)+"<br>Minutes: "+df_edges['Total minutes'].astype(str),
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
        hovertext="Node n°"+df_nodes['Node ID'].astype(str)+"<br>Position: (Lat="+df_nodes['Latitude'].astype(str)+", Lon="+df_nodes['Longitude'].astype(str)+")<br>Total passages: "+df_nodes['Total passages'].astype(str)+"<br>Total minutes: "+df_nodes['Total minutes'].astype(str),
    )
    # GLOBAL SETTINGS:
    fig_update_layout(fig)
    # TITLE:
    fig.update_layout(title_text=f"Default plotly window "+("(day="+str(day) if day is not None and day != "" else "")+", nodes="+str(len(nxgraph.nodes()))+", edges="+str(len(nxgraph.edges()))+")")
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


