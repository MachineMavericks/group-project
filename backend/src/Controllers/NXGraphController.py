# FLASK IMPORTS:
from flask import Flask, Blueprint, render_template, request           # FLASK
from flask_restx import Namespace, Resource                 # REST-X API

# SERVICES=
from src.Services.NXGraphService import *                   # NXGRAPH SERVICE

# SETTINGS=
pickle_filepath = "static/output/graph.pickle"

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
    plotly_heatmap(pickle_path=pickle_filepath, day=request.args.get('day'), component=request.args.get('component'), metric=request.args.get('metric'), output_path='static/output/plotly.html')
    return render_template("filters_plotly.html")
