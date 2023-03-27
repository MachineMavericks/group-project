# FLASK IMPORTS:
from flask_restx import Namespace, Resource

# MODELS=
from src.Models import NXGraph as NXG

# NXGRAPH CONTROLLER/NAMESPACE:
nxgraph_ns = Namespace('nxgraph', description='NXGraph', path='/nxgraph')

@nxgraph_ns.route('/')
class NXGraph(Resource):
    def post(self):
        pass
    def get(self):
        pass
    def delete(self):
        pass
    def put(self):
        pass

@nxgraph_ns.route('/node/<int:id>')
class NXGraphNode(Resource):
    def post(self, id):
        pass
    def get(self, id):
        pass
    def delete(self, id):
        pass
    def put(self, id):
        pass

