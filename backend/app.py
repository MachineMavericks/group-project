# DEFAULT IMPORTS:
import os                                                   # OS
# FLASK IMPORTS:
from flask import Flask, render_template, request           # FLASK
from flask_restx import Api                                 # REST-X API
# MODELS=
from src.Models.Graph import Graph                          # GRAPH
from src.Models.NXGraph import NXGraph                      # NXGRAPH
# CONTROLLERS=
from src.Controllers.GraphController import *               # GRAPH CONTROLLER
from src.Controllers.NXGraphController import *             # NXGRAPH CONTROLLER

# PATHS=
input_dir = "static/input/"
output_dir = "static/output/"
os.mkdir(output_dir) and print("Can't find output directory. Creating one now.") \
    if not os.path.isdir(output_dir) else print("Found existing output directory.")

# GRAPH/NXGRAPH OBJECTS CONSTRUCTION -> SAVE TO PICKLE:
graph = Graph(filepath=input_dir+"railway.csv", output_dir=output_dir, save_csvs=True, save_pickle=True)

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
# ADD NAMESPACES TO API DOC:
api.add_namespace(graph_ns)
api.add_namespace(nxgraph_ns)

# INDEX ROUTE:
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template("base.html")

def main():
    # START THE APP:
    app.run(debug=True)
    # START THE SWAGGER DOC:
    api.init_app(app)
if __name__ == '__main__':
    main()
