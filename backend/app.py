# FLASK IMPORTS:
from flask import Flask, render_template, request           # FLASK
from flask_restx import Api                                 # REST-X API

# PLOTLY IMPORTS:
import plotly.graph_objs as go

# DEFAULT IMPORTS:
import matplotlib.pyplot as plt
import numpy as np
import os

# MODELS=
from src.Models.Graph import  *                             # GRAPH MODEL
from src.Models.NXGraph import *                            # NXGRAPH MODEL
# SERVICES=
# ...
# CONTROLLERS=
from src.Controllers.GraphController import *               # GRAPH CONTROLLER
from src.Controllers.NXGraphController import *             # NXGRAPH CONTROLLER

if not os.path.isdir("static/output"):
    os.mkdir("static/output")
if not os.path.isfile("static/output/nxgraph.gml"):
    nxgraph = NXGraph(Graph("static/input/railway.csv"), dataset_number=1, day=None, save_gml=True, output_path="static/output/nxgraph.gml")

# FLASK APP:
app = Flask(__name__)
# BLUEPRINTS:
app.register_blueprint(nxgraph_bp)


# REST-X API DOC/HANDLER:
api = Api(app,
          version='1.0',
          title='Machine Mavericks ™ - API Documentation',
          description='Official API Documentation for the Machine Mavericks Project.\n\nMade using RestX (Swagger) for Flask.\n\nCourtesy of: Hans Haller.',
          doc='/swagger',
          base_url='/api',
          )
# # ADD NAMESPACES TO API DOC:
api.add_namespace(graph_ns)
api.add_namespace(nxgraph_ns)


@app.route('/index', methods=['GET', 'POST'])
def hey():
    return render_template("base.html")


def main():
    # START THE APP:
    app.run(debug=True)
    # START THE SWAGGER DOC:
    api.init_app(app)
if __name__ == '__main__':
    main()
