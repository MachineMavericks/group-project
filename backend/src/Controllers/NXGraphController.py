# DEFAULT IMPORTS:
import os
from markupsafe import escape
# FLASK IMPORTS:
from flask import Flask, Blueprint, render_template, request, Response  # FLASK
from flask_restx import Namespace, Resource  # REST-X API
# SERVICES=
from src.Services.NXGraphService import *  # NXGRAPH SERVICE

# BLUEPRINT:
nxgraph_bp = Blueprint('nxgraph_bp', __name__)

# PATHS=
input_dir = "static/input/"
output_dir = "static/output/"

# BLUEPRINT ROUTES:
@nxgraph_bp.route('/pickle/<string:railway>/')
def pickle(railway):
    """
    This route is used to build a pickle file from a csv file.
    :param railway: The railway to build the pickle file from.
    :return: A rendered template for the pickle file.
    """
    if railway != "indian" and railway != "chinese":
        return render_template("/error/customdataset_error.html")
    else:
        return render_template("dataset/pickle_load.html")


@nxgraph_bp.route('/progress/<string:railway>/')
def progress(railway):
    """
    This route is used to build a pickle file from a csv file.
    :param railway: The railway to build the pickle file from.
    :return: A rendered template of the progress of the building of the pickle file.
    """
    def build_pickle():
        yield "data:10\n\n"
        Graph(filepath=input_dir + railway + ".csv", output_dir=output_dir, save_csvs=True, save_pickle=True)
        yield "data:100\n\n"
    return Response(build_pickle(), mimetype='text/event-stream')

@nxgraph_bp.route('/default/<string:railway>/', methods=['GET', 'POST'])
def default(railway):
    """
    This route is used to build a default plotly map (from a pickle file).
    It displays the graph on the page, with nodes and edges and basic information (total minutes, passages, etc).
    :param railway: The railway to be analyzed.
    :return: A rendered template for the default plotly graph.
    """
    if os.path.isfile("static/output/" + railway + ".pickle"):
        plotly_default(pickle_path="static/output/" + railway + ".pickle", output_path='static/output/plotly.html', day=request.args.get('day'))
        return render_template("default_plotly.html")
    else:
        raise Exception("No pickle file found for the specified railway.")
        return render_template("error/pickle_error.html")


@nxgraph_bp.route('/heatmap/<string:railway>/', methods=['GET', 'POST'])
def heatmap(railway):
    """
    This route is used to build a heatmap plotly map (from a pickle file).
    It displays an heatmap of the graph on the page, with nodes and edges and basic information (total minutes, passages, etc).
    The component and metric can be specified in the url.
    :param railway: The railway to be analyzed.
    :return:
    """
    if os.path.isfile("static/output/" + railway + ".pickle"):
        plotly_heatmap(pickle_path="static/output/" + railway + ".pickle", day=request.args.get('day'), component=request.args.get('component'),
                          metric=request.args.get('metric'), output_path='static/output/plotly.html')
        return render_template("heatmap_plotly.html")
    else:
        raise Exception("No pickle file found for the specified railway.")
        return render_template("error/pickle_error.html")


@nxgraph_bp.route('/resilience/<string:railway>/', methods=['GET', 'POST'])
def resilience(railway):
    """
    This route is used to build a resilience plotly map (from a pickle file) and display it on the page.
    The strategy, component, metric, day, fraction and smallworld can be specified in the url.
    It displays the graph on the page, with nodes and edges and basic information (total minutes, passages, etc).
    It also displays the nodes that are removed from the graph, and their value for the specified metric that was
    used to sort them.
    :param railway: The railway to be analyzed.
    :return: A rendered template for the resilience plotly graph.
    """
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
    """
    This route is used to build a clustering plotly map (from a pickle file) and display it on the page.
    The algorithm, weight, day and adv_legend can be specified in the url.
    It displays the graph on the page, with nodes and edges and basic information (total minutes, passages, etc).
    It also displays the clusters that were found by the algorithm, and for each node, the cluster it belongs to.
    :param railway:
    :return:
    """
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
    """
    This route is used to build the smallworld graphs (from a pickle file) and display them on the page.
    The day can be specified in the url.
    It gives information about the smallworldness of the graph, and gives information about the clustering coefficient,
    the average shortest path and the smallworldness coefficient, and more.
    :param railway: The railway to be analyzed.
    :return: A rendered template for the smallworld plotly graph.
    """
    if os.path.isfile("static/output/" + railway + ".pickle"):
        plotly_small_world(pickle_path="static/output/" + railway + ".pickle", output_path='static/output/plotly.html',
                           day=request.args.get('day'))
        return render_template("smallworld_plotly.html")
    else:
        raise Exception("No pickle file found for the specified railway.")
        return render_template("error/pickle_error.html")


@nxgraph_bp.route('/shortest_path/<string:railway>/', methods=['GET', 'POST'])
def shortest_path(railway):
    """
    This route is used to build the temporal shortest path plotly map (from a pickle file) and display it on the page.
    The day, departure node, destination node and departure time can be specified in the url.
    It displays the graph on the page, with nodes and edges and basic information (total minutes, passages, etc).
    It also displays the shortest path between the departure and destination nodes, and the time it takes to travel,
    as well as the number of passages and the trains that are used.
    Additionaly, it also draws the basic un-temporal shortest path on the graph.
    :param railway: The railway to be analyzed.
    :return: A rendered template for the temporal shortest path plotly graph.
    """
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
    """
    This route is used to build the centrality graphs (from a pickle file) and display theù on the page.
    The day can be specified in the url. It displays the graph on the page, with nodes and edges and basic information.
    It also displays the centrality of each node, and the nodes that have the highest centrality.
    :param railway: The railway to be analyzed.
    :return: A rendered template for the centrality plotly graph.
    """
    if os.path.isfile("static/output/" + railway + ".pickle"):
        plotly_centrality(pickle_path="static/output/" + railway + ".pickle", output_path='static/output/plotly.html',
                           day=request.args.get('day'))
        return render_template("centrality_plotly.html")
    else:
        raise Exception("No pickle file found for the specified railway.")
        return render_template("error/pickle_error.html")

@nxgraph_bp.route('/correlation/<string:railway>/', methods=['GET', 'POST'])
def correlation(railway):
    """
    This route is used to build the correlation graphs (from a pickle file) and display them on the page.
    The day and neighbor can be specified in the url. It displays the graph on the page, with nodes and edges.
    It also displays the correlation between each node and its neighbors.
    :param railway: The railway to be analyzed.
    :param railway:
    :return:
    """
    if os.path.isfile("static/output/" + railway + ".pickle"):
        plotly_correlation(pickle_path="static/output/" + railway + ".pickle", output_path='static/output/plotly.html',
                           day=request.args.get('day'), neighbor=request.args.get('neighbor'))
        return render_template("correlation_plotly.html")
    else:
        raise Exception("No pickle file found for the specified railway.")
        return render_template("error/pickle_error.html")
