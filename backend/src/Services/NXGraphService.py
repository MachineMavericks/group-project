from src.Models import NXGraph as NXG
from src.Models import Graph as G

graph = G.Graph('resources/data/input/railway.csv')
nxgraph = NXG.nxgraph_loader(graph)

def nodes_whose_centrality_degree_is_greater_than(graph, degree):
    # TODO.
    # return nodes[]
    pass