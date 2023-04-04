# FLASK IMPORTS:
from flask import Flask, Blueprint, render_template, request           # FLASK
from flask_restx import Namespace, Resource                 # REST-X API
# DEFAULT IMPORTS:
import networkx as nx
# SERVICES=
from src.Services.NXGraphService import *                   # NXGRAPH SERVICE

# NXGRAPH CONTROLLER/NAMESPACE:
nxgraph_ns = Namespace('nxgraph', description='NXGraph', path='/api/nxgraph')

# BLUEPRINT:
nxgraph_bp = Blueprint('nxgraph_bp', __name__, template_folder='templates')

# API ROUTES:
@nxgraph_ns.route('/')
class NXGraphNS(Resource):
    def post(self):
        pass

    def get(self):
        pass

    def delete(self):
        pass

    def put(self):
        pass

@nxgraph_ns.route('/node/<int:id>')
class NXGraphNodeNS(Resource):
    def post(self, id):
        pass

    def get(self, id):
        pass

    def delete(self, id):
        pass

    def put(self, id):
        pass

# BLUEPRINT ROUTES:
@nxgraph_bp.route('/heatmap', methods=['GET', 'POST'])
def heatmap():
    day = request.args.get('day')
    component = request.args.get('component')
    metric = request.args.get('metric')
    nxgraph = nx.read_gml('static/output/nxgraph.gml')
    plotly_heatmap(nxgraph, day=day, component=component, metric=metric, output_path='static/output/plotly.html')
    return render_template("base_plotly.html")
