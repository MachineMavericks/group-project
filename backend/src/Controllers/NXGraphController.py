# DEFAULT IMPORTS:
import os
from markupsafe import escape

# FLASK IMPORTS:
from flask import Flask, Blueprint, render_template, request  # FLASK
from flask_restx import Namespace, Resource  # REST-X API

# SERVICES=
from src.Services.NXGraphService import *  # NXGRAPH SERVICE


# NXGRAPH CONTROLLER/NAMESPACE:
nxgraph_ns = Namespace('nxgraph', description='NXGraph', path='/api/nxgraph')

# BLUEPRINT:
nxgraph_bp = Blueprint('nxgraph_bp', __name__)


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
@nxgraph_bp.route('/default/<string:railway>/', methods=['GET', 'POST'])
def default(railway):
    if os.path.isfile("static/output/" + railway + ".pickle"):
        plotly_default(pickle_path="static/output/" + railway + ".pickle", output_path='static/output/plotly.html', day=request.args.get('day'))
        return render_template("default_plotly.html")
    else:
        raise Exception("No pickle file found for the specified railway.")
        return render_template("error/pickle_error.html")


@nxgraph_bp.route('/heatmap/<string:railway>/', methods=['GET', 'POST'])
def heatmap(railway):
    if os.path.isfile("static/output/" + railway + ".pickle"):
        if plotly_heatmap(pickle_path="static/output/" + railway + ".pickle", day=request.args.get('day'), component=request.args.get('component'),
                       metric=request.args.get('metric'), output_path='static/output/plotly.html'):
            return render_template("heatmap_plotly.html")
        else:
            return render_template("error/custom_error.html")
    else:
        raise Exception("No pickle file found for the specified railway.")
        return render_template("error/pickle_error.html")


@nxgraph_bp.route('/resiliency/<string:railway>/', methods=['GET', 'POST'])
def resilience(railway):
    if os.path.isfile("static/output/" + railway + ".pickle"):
        if plotly_resilience(pickle_path="static/output/" + railway + ".pickle", output_path='static/output/plotly.html',
                      strategy=request.args.get('type'),
                      component=request.args.get('component'), metric=request.args.get('metric'),
                      day=request.args.get('day'), fraction=request.args.get('fraction')):
            return render_template("resilience_plotly.html")
        else:
            return render_template("error/custom_error.html")
    else:
        raise Exception("No pickle file found for the specified railway.")
        return render_template("error/pickle_error.html")


@nxgraph_bp.route('/clustering/<string:railway>/', methods=['GET', 'POST'])
def clustering(railway):
    if os.path.isfile("static/output/" + railway + ".pickle"):
        plotly_clustering(pickle_path="static/output/" + railway + ".pickle", output_path='static/output/plotly.html',
                          algorithm=request.args.get('algorithm'),
                          weight=request.args.get('weight'),
                          day=request.args.get('day'),
                          adv_legend=True if request.args.get('adv_legend') == "true" else False)
        return render_template("clustering_plotly.html")
    else:
        raise Exception("No pickle file found for the specified railway.")
        return render_template("error/pickle_error.html")

@nxgraph_bp.route('/smallworld/<string:railway>/', methods=['GET', 'POST'])
def smallworld(railway):
    if os.path.isfile("static/output/" + railway + ".pickle"):
        plotly_small_world(pickle_path="static/output/" + railway + ".pickle", output_path='static/output/plotly.html',
                           day=request.args.get('day'))
        return render_template("smallworld_plotly.html")
    else:
        raise Exception("No pickle file found for the specified railway.")
        return render_template("error/pickle_error.html")
