# FLASK IMPORTS:
from flask import Flask                                     # FLASK
from flask_restx import Api                                 # REST-X API
# DASH IMPORTS:
import dash
from dash import dcc
from dash import html
# PLOTLY IMPORTS:
import plotly.graph_objs as go
import plotly.express as px

# MODELS=
from src.Models.Graph import  *                             # GRAPH MODEL
from src.Models.NXGraph import *                            # NXGRAPH MODEL
# SERVICES=
# ...
# CONTROLLERS=
from src.Controllers.GraphController import *               # GRAPH CONTROLLER
from src.Controllers.NXGraphController import *             # NXGRAPH CONTROLLER

# OBJECTS=
# graph = G.Graph('resources/data/input/railway.csv')
# nxgraph = NXG.nxgraph_loader(graph)
nxgraph = NXG.gml_file_to_nx('resources/data/output/nxgraph.gml')

with open('resources/data/input/china.json') as f:
    china_geojson = json.load(f)

# Create map figure
fig = px.choropleth_mapbox(
    pd.DataFrame(),
    geojson=china_geojson,
    locations=[],
    color=[],
    opacity=0.0,
    center={"lat": 35.8617, "lon": 104.1954},
    mapbox_style="carto-positron",
    zoom=3
)

# Add nodes to map figure
i = 0
for node in nxgraph.nodes.data():
    fig.add_trace(
        go.Scattermapbox(
            lat=[node[1]['lat']],
            lon=[node[1]['lon']],
            mode='markers',
            marker=dict(size=5, color='blue'),
            text=node[0],
            name='Node'
        )
    )
    i += 1
    if i > 100:
        break

# Add edges to map figure
i = 0
for edge in nxgraph.edges.data():
    fig.add_trace(
        go.Scattermapbox(
            # Use fromLat, fromLon, destLat, destLon to draw edges
            lat=[edge[2]['fromLat'], edge[2]['destLat']],
            lon=[edge[2]['fromLon'], edge[2]['destLon']],
            mode='lines',
            line=dict(width=1, color='black'),
            name='Edge'
        )
    )
    i += 1
    if i > 100:
        break


# Set up Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(
        id='graph',
        figure=fig
    )
])



# app = Flask(__name__)

# REST-X API DOC/HANDLER:
# api = Api(app,
#           version='1.0',
#           title='Machine Mavericks ™ - API Documentation',
#           description='Official API Documentation for the Machine Mavericks Project.\n\nMade using RestX (Swagger) for Flask.\n\nCourtesy of: Hans Haller.',
#           doc='/swagger',
#           )
# ADD NAMESPACES TO API DOC:
# api.add_namespace(graph_ns)
# api.add_namespace(nxgraph_ns)


def main():
    # START THE APP:
    app.run_server(debug=True)
    # START THE SWAGGER DOC:
    # api.init_app(app)
if __name__ == '__main__':
    main()