# FLASK IMPORTS:
from flask import Flask, render_template, request           # FLASK
from flask_restx import Namespace, Resource                 # REST-X API

# MODELS=
from src.Models import Graph as G

# GRAPH CONTROLLER/NAMESPACE:
graph_ns = Namespace('graph', description='Graph', path='/api/graph')

@graph_ns.route('/')
class Graph(Resource):
    def get(self):
        pass

    def put(self):
        pass

    def delete(self):
        pass

    def post(self):
        pass

@graph_ns.route('/node/<int:id>')
class GraphNode(Resource):
    def get(self, id):
        pass

    def put(self, id):
        pass

    def delete(self, id):
        pass

    def post(self, id):
        pass
