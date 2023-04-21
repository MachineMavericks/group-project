# DEFAULT IMPORTS:
import os
from markupsafe import escape

# FLASK IMPORTS:
from flask import Flask, Blueprint, render_template, request, Response  # FLASK
from flask_restx import Namespace, Resource  # REST-X API

# SERVICES=
from src.Services.NXGraphService import *  # NXGRAPH SERVICE


# NXGRAPH CONTROLLER/NAMESPACE:
nxgraph_ns = Namespace('nxgraph', description='NXGraph', path='/api/nxgraph')

# BLUEPRINT:
nxgraph_bp = Blueprint('nxgraph_bp', __name__)

# PATHS=
input_dir = "static/input/"
output_dir = "static/output/"

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
@nxgraph_bp.route('/pickle/<string:railway>/')
def pickle(railway):
    if railway != "indian" and railway != "chinese":
        return render_template("/error/customdataset_error.html")
    else:
        return render_template("dataset/pickle_load.html")


@nxgraph_bp.route('/progress/<string:railway>/')
def progress(railway):
    def build_pickle():
        yield "data:10\n\n"
        Graph(filepath=input_dir + railway + ".csv", output_dir=output_dir, save_csvs=True, save_pickle=True)
        yield "data:100\n\n"
    return Response(build_pickle(), mimetype='text/event-stream')

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
        plotly_heatmap(pickle_path="static/output/" + railway + ".pickle", day=request.args.get('day'), component=request.args.get('component'),
                          metric=request.args.get('metric'), output_path='static/output/plotly.html')
        return render_template("heatmap_plotly.html")
    else:
        raise Exception("No pickle file found for the specified railway.")
        return render_template("error/pickle_error.html")


@nxgraph_bp.route('/resilience/<string:railway>/', methods=['GET', 'POST'])
def resilience(railway):
    if os.path.isfile("static/output/" + railway + ".pickle"):
        plotly_resilience(pickle_path="static/output/" + railway + ".pickle", output_path='static/output/plotly.html',
                          strategy=request.args.get('strategy'), component=request.args.get('component'),
                          metric=request.args.get('metric'), day=request.args.get('day'),
                          fraction=request.args.get('fraction'), smallworld=request.args.get('smallworld'))
        return render_template("resilience_plotly.html")
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


@nxgraph_bp.route('/shortest_path/<string:railway>/', methods=['GET', 'POST'])
def shortest_path(railway):
    if os.path.isfile("static/output/" + railway + ".pickle"):
        plotly_shortest_path(pickle_path="static/output/" + railway + ".pickle", output_path='static/output/plotly.html',
                             day=request.args.get('day'),
                             dep_time=request.args.get('departure'),
                             start=request.args.get('from'),
                             end=request.args.get('destination'))
        return render_template("shortest_path_plotly.html")
    else:
        raise Exception("No pickle file found for the specified railway.")
        return render_template("error/pickle_error.html")

@nxgraph_bp.route('/centrality/<string:railway>/', methods=['GET', 'POST'])
def centrality(railway):
    if os.path.isfile("static/output/" + railway + ".pickle"):
        plotly_centrality(pickle_path="static/output/" + railway + ".pickle", output_path='static/output/plotly.html',
                           day=request.args.get('day'))
        return render_template("centrality_plotly.html")
    else:
        raise Exception("No pickle file found for the specified railway.")
        return render_template("error/pickle_error.html")
