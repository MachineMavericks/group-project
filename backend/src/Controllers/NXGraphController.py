# FLASK IMPORTS:
from flask import Flask, Blueprint, render_template, request  # FLASK
from flask_restx import Namespace, Resource  # REST-X API

# SERVICES=
from src.Services.NXGraphService import *  # NXGRAPH SERVICE

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
@nxgraph_bp.route('/default', methods=['GET', 'POST'])
def default():
    plotly_default(pickle_path=pickle_filepath, output_path='static/output/plotly.html', day=request.args.get('day'))
    return render_template("default_plotly.html")


@nxgraph_bp.route('/heatmap', methods=['GET', 'POST'])
def heatmap():
    plotly_heatmap(pickle_path=pickle_filepath, day=request.args.get('day'), component=request.args.get('component'),
                   metric=request.args.get('metric'), output_path='static/output/plotly.html')
    return render_template("heatmap_plotly.html")


@nxgraph_bp.route('/resilience', methods=['GET', 'POST'])
def resilience():
    plotly_resilience(pickle_path=pickle_filepath, output_path='static/output/plotly.html',
                      strategy=request.args.get('type'),
                      component=request.args.get('component'), metric=request.args.get('metric'),
                      day=request.args.get('day'), fraction=request.args.get('fraction'))
    return render_template("resilience_plotly.html")


@nxgraph_bp.route('/clustering', methods=['GET', 'POST'])
def clustering():
    plotly_clustering(pickle_path=pickle_filepath, output_path='static/output/plotly.html',
                      algorithm=request.args.get('algorithm'),
                      weight=request.args.get('weight'),
                      day=request.args.get('day'))
    return render_template("clustering_plotly.html")

@nxgraph_bp.route('/smallworld', methods=['GET', 'POST'])
def histogram():
    plotly_small_world(pickle_path=pickle_filepath, output_path='static/output/plotly.html',
                       day=request.args.get('day'), nb_start_node=request.args.get('nb_start'))
    return render_template("smallworld_plotly.html")
