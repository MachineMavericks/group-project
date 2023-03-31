# FLASK IMPORTS:
from flask import Flask  # FLASK
from flask_restx import Api  # REST-X API
# PLOTLY IMPORTS:
import plotly.graph_objs as go

# MODELS=
from src.Models.Graph import *  # GRAPH MODEL
from src.Models.NXGraph import *  # NXGRAPH MODEL
# SERVICES=
# ...
# CONTROLLERS=
from src.Controllers.GraphController import *  # GRAPH CONTROLLER
from src.Controllers.NXGraphController import *  # NXGRAPH CONTROLLER

# OBJECTS=
# graph = G.Graph('resources/data/input/railway.csv')
# nxgraph = NXG.nxgraph_loader(graph)
nxgraph = NXG.gml_file_to_nx('resources/data/output/nxgraph.gml')
with open('resources/data/input/china.json') as f:
    china_geojson = json.load(f)

# FLASK APP:
app = Flask(__name__)
# REST-X API DOC/HANDLER:
api = Api(app,
          version='1.0',
          title='Machine Mavericks ™ - API Documentation',
          description='Official API Documentation for the Machine Mavericks Project.\n\nMade using RestX (Swagger) '
                      'for Flask.\n\nCourtesy of: Hans Haller.',
          doc='/swagger',
          )
# ADD NAMESPACES TO API DOC:
api.add_namespace(graph_ns)
api.add_namespace(nxgraph_ns)

# PLOT GRAPH NODES AND EDGES USING PLOTLY (SCATTERGEO):
fig = go.Figure()
# NODES:
for node in nxgraph.nodes(data=True):
    fig.add_trace(go.Scattergeo(
        lon=[node[1]['lon']],
        lat=[node[1]['lat']],
        mode='markers',
        marker=dict(
            size=5,
            color='rgb(255, 0, 0)',
            line=dict(
                width=3,
                color='rgba(68, 68, 68, 0)'
            )
        ),
        text=node[0],
    ))
# EDGES:
for edge in nxgraph.edges(data=True):
    fig.add_trace(go.Scattergeo(
        lon=[edge[2]['fromLon'], edge[2]['destLon'], None],
        lat=[edge[2]['fromLat'], edge[2]['destLat'], None],
        mode='lines',
        line=dict(
            width=1,
            color='red',
        ),
        text=edge[0],
    ))
# SETTINGS=
fig.update_layout(showlegend=False)
# ADD FIG TO FLASK APP:
app.add_url_rule('/plotly', 'plotly', lambda: fig.to_html(full_html=False, include_plotlyjs='cdn'))

# USE KMEANS TO CLUSTER THE GRAPH NODES - THEN PLOT THEM USING PLOTLY (SCATTERGEO):
from sklearn.cluster import KMeans
pos = {node[0]: (node[1]['lon'], node[1]['lat']) for node in nxgraph.nodes(data=True)}
kmeans = KMeans(n_clusters=10, random_state=0).fit(nxgraph.nodes.data())
clusters = kmeans.predict(list(pos.values()))
fig2 = go.Figure()
for cluster in range(0, 10):
    fig2.add_trace(go.Scattergeo(
        lon=[pos[node[0]][0] for node in nxgraph.nodes.data() if clusters[node[0]] == cluster],
        lat=[pos[node[0]][1] for node in nxgraph.nodes.data() if clusters[node[0]] == cluster],
        mode='markers',
        marker=dict(
            size=5,
            color='rgb(255, 0, 0)',
            line=dict(
                width=3,
                color='rgba(68, 68, 68, 0)'
            )
        ),
        text=node[0],
    ))
# SETTINGS=
fig2.update_layout(showlegend=False)
# ADD FIG TO FLASK APP:
app.add_url_rule('/plotly2', 'plotly2', lambda: fig2.to_html(full_html=False, include_plotlyjs='cdn'))


def main():
    # START THE APP:
    app.run(debug=True)
    # START THE SWAGGER DOC:
    api.init_app(app)


if __name__ == '__main__':
    main()
